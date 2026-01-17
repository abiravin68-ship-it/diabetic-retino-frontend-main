import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Alert,
  Badge,
  Modal,
  Spinner,
  Form,
} from "react-bootstrap";
import { useDropzone } from "react-dropzone";
import {
  FaUpload,
  FaImage,
  FaBrain,
  FaCheckCircle,
  FaShieldAlt,
  FaInfoCircle,
} from "react-icons/fa";
import axios from "axios";
import "./index.css";

const MAX_FILE_BYTES = 5 * 1024 * 1024;
const HEALTH_POLL_MS = 5000;

const RAW_API_BASE = String(import.meta.env.VITE_API_BASE_URL || "").trim();
const HAS_EXPLICIT_API_URL = Boolean(RAW_API_BASE);

const API_BASE = (() => {
  const env = RAW_API_BASE;
  if (env) return env.replace(/\/+$/, "");
  if (import.meta.env.DEV) return "http://localhost:8000";
  return "";
})();

const HEALTH_PATHS = ["/api/health"];
const PREDICT_PATHS = ["/api/predict"];
const MODEL_INFO_PATH = "/api/model-info";
const PRIVACY_NOTICE_PATH = "/api/privacy-notice";

function normalizePercent(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return null;
  if (x >= 0 && x <= 1) return x * 100;
  return x;
}

function toSafeClassLabel(label) {
  if (!label) return "";
  return String(label).replace(/_/g, " ");
}

function extractServerError(err) {
  const status = err?.response?.status;
  const data = err?.response?.data;

  const serverMsg = data?.error || data?.message;
  if (serverMsg) return { status, message: serverMsg };

  if (status) return { status, message: `Request failed with status ${status}.` };

  const msg =
    err?.message ||
    "Failed to connect to the server. Ensure the backend is running and CORS/proxy is configured.";
  return { status: null, message: msg };
}


function normalizeApiResponse(data) {
  if (!data || typeof data !== "object") {
    return { success: false, error: "Invalid server response." };
  }

  if (data.success === false) {
    return {
      success: false,
      error: data.error || data.message || "Prediction failed.",
      dev_mode: Boolean(data.dev_mode),
      model_load_error: data.model_load_error || null,
    };
  }

  let className = "";
  let confidence = 0;
  let description = "";

  if (data.prediction && typeof data.prediction === "object") {
    className = data.prediction.class || data.prediction.label || "";
    confidence =
      normalizePercent(
        data.prediction.confidence ?? data.confidence ?? data.prediction.score ?? 0
      ) ?? 0;
    description = data.prediction.description || data.description || "";
  } else {
    className =
      (typeof data.prediction === "string" && data.prediction) ||
      data.prediction_label ||
      data.class ||
      "";
    confidence = normalizePercent(data.confidence ?? 0) ?? 0;
    description = data.description || "";
  }

  const probsRaw =
    data.all_probabilities ||
    data.probabilities ||
    data.compat?.probabilities ||
    {};

  const all_probabilities = Object.fromEntries(
    Object.entries(probsRaw).map(([k, v]) => [k, normalizePercent(v) ?? 0])
  );

  return {
    success: true,
    session_id: data.session_id || null,
    prediction: { class: className, confidence, description },
    all_probabilities,
    gradcam_image: data.gradcam_image || null,
    security: data.security || null,
    storage: data.storage || null,
    elapsed_ms: typeof data.elapsed_ms === "number" ? data.elapsed_ms : null,
  };
}

function getSeverityGradient(className) {
  const gradients = {
    No_DR: "linear-gradient(135deg, #28a745 0%, #20c997 100%)",
    Mild: "linear-gradient(135deg, #ffc107 0%, #ffca2c 100%)",
    Moderate: "linear-gradient(135deg, #fd7e14 0%, #ff8c42 100%)",
    Severe: "linear-gradient(135deg, #dc3545 0%, #e4606d 100%)",
    Proliferative_DR: "linear-gradient(135deg, #6f42c1 0%, #8c68cd 100%)",
  };
  return (
    gradients[className] ||
    "linear-gradient(135deg, #6c757d 0%, #868e96 100%)"
  );
}

function joinUrl(base, path) {
  if (!base) return path;
  if (!path) return base;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}

export default function App() {
  const [backendHealth, setBackendHealth] = useState(null);
  const [backendHealthError, setBackendHealthError] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [wantGradcam, setWantGradcam] = useState(false);

  const [cooldownUntil, setCooldownUntil] = useState(0);
  const isCoolingDown = cooldownUntil > Date.now();
  const cooldownSecondsLeft = Math.max(
    0,
    Math.ceil((cooldownUntil - Date.now()) / 1000)
  );

  const [showModelInfo, setShowModelInfo] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(false);
  const [modelInfoError, setModelInfoError] = useState("");

  const [showPrivacy, setShowPrivacy] = useState(false);
  const [privacyInfo, setPrivacyInfo] = useState(null);
  const [privacyLoading, setPrivacyLoading] = useState(false);
  const [privacyError, setPrivacyError] = useState("");

  const probs = prediction?.all_probabilities || {};
  const probEntries = useMemo(
    () => Object.entries(probs).sort((a, b) => (b[1] || 0) - (a[1] || 0)),
    [probs]
  );

  const apiModeLabel = useMemo(() => {
    if (import.meta.env.DEV) return "Local";
    if (HAS_EXPLICIT_API_URL) return "Direct (Render URL)";
    return "Proxy (same-origin)";
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    const fetchHealth = async () => {
      if (import.meta.env.PROD && !API_BASE) {
        setBackendHealth(null);
        setBackendHealthError(
          "VITE_API_BASE_URL is not set. Add it in Vercel (Production + Preview) to your Render backend URL, then redeploy."
        );
        return;
      }

      try {
        let lastErr = null;
        let res = null;

        for (const path of HEALTH_PATHS) {
          try {
            res = await axios.get(joinUrl(API_BASE, path), {
              timeout: 20000,
              withCredentials: false,
            });
            break;
          } catch (e) {
            lastErr = e;
          }
        }

        if (!res) throw lastErr;
        if (cancelled) return;

        setBackendHealth(res.data);
        setBackendHealthError("");

        const isLoading = Boolean(res.data?.model_loading);
        const isLoaded = Boolean(res.data?.model_loaded);

        if (isLoading && !isLoaded) {
          timer = window.setTimeout(fetchHealth, HEALTH_POLL_MS);
        }
      } catch (e) {
        if (cancelled) return;
        const { message } = extractServerError(e);
        setBackendHealth(null);
        setBackendHealthError(message);
      }
    };

    fetchHealth();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles && acceptedFiles[0];
    if (!file) return;

    setError("");
    setPrediction(null);

    const extOk = /\.(png|jpg|jpeg)$/i.test(file.name || "");
    if (!extOk) {
      setSelectedFile(null);
      setPreview((prevUrl) => {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return null;
      });
      setError("Invalid file type. Please upload PNG, JPG, or JPEG.");
      return;
    }

    if (file.size > MAX_FILE_BYTES) {
      setSelectedFile(null);
      setPreview((prevUrl) => {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return null;
      });
      setError("File too large. Maximum allowed size is 5 MB.");
      return;
    }

    setSelectedFile(file);

    const nextUrl = URL.createObjectURL(file);
    setPreview((prevUrl) => {
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      return nextUrl;
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg"] },
    multiple: false,
    disabled: loading,
  });

  const backendNotReady =
    !!backendHealth && (!backendHealth.model_loaded || backendHealth.model_loading);

  async function handleAnalyze() {
    if (loading) return;

    if (isCoolingDown) {
      setError(`Rate limit exceeded. Try again in ${cooldownSecondsLeft}s.`);
      return;
    }

    if (backendHealth?.model_loading) {
      setError("Model is still loading on the backend. Please wait a few seconds and try again.");
      return;
    }

    if (backendHealth && !backendHealth.model_loaded) {
      setError("Backend model is not ready. Please check the backend /api/health status.");
      return;
    }

    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      let lastErr = null;

      const predictPaths = wantGradcam
        ? ["/api/predict?gradcam=1", ...PREDICT_PATHS]
        : PREDICT_PATHS;

      for (const path of predictPaths) {
        try {
          const res = await axios.post(joinUrl(API_BASE, path), formData, {
            timeout: 300000,
            withCredentials: false,
          });

          const normalized = normalizeApiResponse(res.data);
          if (!normalized.success) {
            setError(normalized.error || "Prediction failed.");
            return;
          }

          setPrediction(normalized);
          return;
        } catch (e) {
          lastErr = e;
        }
      }

      throw lastErr;
    } catch (e) {
      const { status, message } = extractServerError(e);

      if (status === 413) {
        setError("File too large. Maximum allowed size is 5 MB.");
      } else if (status === 429) {
        const retryAfter = Number(e?.response?.headers?.["retry-after"]);
        const waitMs = Number.isFinite(retryAfter) ? retryAfter * 1000 : 60_000;
        setCooldownUntil(Date.now() + waitMs);
        setError(`Rate limit exceeded. Try again in ${Math.ceil(waitMs / 1000)}s.`);
      } else if (status === 503) {
        const retryAfter = Number(e?.response?.headers?.["retry-after"]);
        const waitMs = Number.isFinite(retryAfter) ? retryAfter * 1000 : 5_000;
        setError(message || "Backend unavailable (503).");
        setCooldownUntil(Date.now() + waitMs);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setSelectedFile(null);
    setPrediction(null);
    setError("");
    setPreview((prevUrl) => {
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      return null;
    });
  }

  async function openModelInfo() {
    setShowModelInfo(true);
    if (modelInfo || modelInfoLoading) return;

    setModelInfoLoading(true);
    setModelInfoError("");

    try {
      const res = await axios.get(joinUrl(API_BASE, MODEL_INFO_PATH), {
        timeout: 20000,
        withCredentials: false,
      });

      setModelInfo(res.data);
    } catch (e) {
      const { message } = extractServerError(e);
      setModelInfoError(message);
    } finally {
      setModelInfoLoading(false);
    }
  }

  async function openPrivacyNotice() {
    setShowPrivacy(true);
    if (privacyInfo || privacyLoading) return;

    setPrivacyLoading(true);
    setPrivacyError("");

    try {
      const res = await axios.get(joinUrl(API_BASE, PRIVACY_NOTICE_PATH), {
        timeout: 20000,
        withCredentials: false,
      });

      setPrivacyInfo(res.data);
    } catch (e) {
      const { message } = extractServerError(e);
      setPrivacyError(message);
    } finally {
      setPrivacyLoading(false);
    }
  }

  const safeClass = prediction?.prediction?.class || "";
  const safeConfidence =
    typeof prediction?.prediction?.confidence === "number"
      ? prediction.prediction.confidence
      : 0;
  const safeDesc = prediction?.prediction?.description || "";

  const backendBadge = backendHealth ? (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
      {backendHealth.model_loaded ? (
        <>
          <Badge bg="success" style={{ fontSize: 13 }}>
            Backend Ready
          </Badge>
          <span style={{ fontSize: 12, opacity: 0.9 }}>Model loaded</span>
        </>
      ) : backendHealth.model_loading ? (
        <>
          <Badge bg="info" style={{ fontSize: 13 }}>
            Model Loading
          </Badge>
          <span style={{ fontSize: 12, opacity: 0.9 }}>Warming up model… please wait</span>
        </>
      ) : (
        <>
          <Badge bg="warning" style={{ fontSize: 13 }}>
            Backend No Model
          </Badge>
          <span style={{ fontSize: 12, opacity: 0.9 }}>Model not loaded</span>
        </>
      )}
    </div>
  ) : backendHealthError ? (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
      <Badge bg="danger" style={{ fontSize: 13 }}>
        Backend Offline
      </Badge>
      <span style={{ fontSize: 12, opacity: 0.9 }}>{backendHealthError}</span>
      <Badge bg="secondary" style={{ fontSize: 12 }}>
        API: {apiModeLabel}
      </Badge>
    </div>
  ) : (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
      <Badge bg="secondary" style={{ fontSize: 13 }}>
        Checking Backend
      </Badge>
      <Badge bg="secondary" style={{ fontSize: 12 }}>
        API: {apiModeLabel}
      </Badge>
    </div>
  );

  const uploadSection = preview ? (
    <div className="preview-container">
      <img src={preview} alt="Preview" className="preview-image-modern" />

      <div className="mt-3" style={{ display: "flex", justifyContent: "center" }}>
        <Form.Check
          type="switch"
          id="gradcam-switch"
          label="Generate Grad-CAM"
          checked={wantGradcam}
          onChange={(e) => setWantGradcam(e.target.checked)}
          disabled={loading}
        />
      </div>

      <div className="button-group mt-4">
        <Button
          variant="primary"
          size="lg"
          className="btn-modern btn-analyze"
          onClick={handleAnalyze}
          disabled={loading || isCoolingDown || backendNotReady}
        >
          <FaBrain className="me-2" />
          {backendHealth?.model_loading
            ? "Model Loading..."
            : isCoolingDown
            ? `Wait ${cooldownSecondsLeft}s`
            : loading
            ? "Analyzing..."
            : "Analyze Now"}
        </Button>

        <Button
          variant="outline-light"
          size="lg"
          className="btn-modern"
          onClick={handleClear}
          disabled={loading}
        >
          Upload New
        </Button>
      </div>
    </div>
  ) : (
    <div {...getRootProps()} className={`dropzone-modern ${isDragActive ? "active" : ""}`}>
      <input {...getInputProps()} />
      <div className="dropzone-content">
        <FaUpload className="upload-icon" />
        <h4>Drop your image here</h4>
        <p>or click to browse</p>
        <small className="text-muted">Supports: PNG, JPG, JPEG (max 5 MB)</small>
      </div>
    </div>
  );

  const securityFlags = prediction?.security || null;
  const storageFlags = prediction?.storage || null;

  const showSecurity =
    securityFlags &&
    typeof securityFlags === "object" &&
    ["encrypted", "anonymized", "gdpr_compliant", "pdpa_compliant"].some((k) => k in securityFlags);

  const storedUpload =
    storageFlags && typeof storageFlags === "object" ? Boolean(storageFlags.stored_upload) : null;
  const encryptedUploadId =
    storageFlags && typeof storageFlags === "object" ? storageFlags.encrypted_upload_id : null;

  const resultsContent = prediction ? (
    <div className="results-content">
      <div className="result-header">
        <FaCheckCircle className="success-icon" />
        <h3>Diagnosis Complete</h3>
      </div>

      {prediction.session_id ? (
        <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 10 }}>
          Session ID:{" "}
          <span style={{ fontFamily: "monospace" }}>{prediction.session_id}</span>
        </div>
      ) : null}

      <div className="diagnosis-card" style={{ background: getSeverityGradient(safeClass) }}>
        <div className="diagnosis-class">{toSafeClassLabel(safeClass)}</div>
        <div className="diagnosis-confidence">{safeConfidence.toFixed(1)}% Confidence</div>
        <div className="diagnosis-description">{safeDesc}</div>
      </div>

      <div className="probability-section">
        <h5 className="mb-3">Probability Distribution</h5>

        {probEntries.length ? (
          probEntries.map(([cls, p]) => {
            const pct = Math.max(0, Math.min(100, typeof p === "number" ? p : 0));
            return (
              <div key={cls} className="prob-item">
                <div className="prob-header">
                  <span className="prob-label">{toSafeClassLabel(cls)}</span>
                  <span className="prob-value">{pct.toFixed(1)}%</span>
                </div>
                <div className="prob-bar-container">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${pct}%`, background: getSeverityGradient(cls) }}
                  />
                </div>
              </div>
            );
          })
        ) : (
          <p className="text-muted mb-0">No probability distribution returned by the server.</p>
        )}
      </div>

      {prediction.gradcam_image ? (
        <div className="mt-4">
          <h5 className="mb-2">Grad-CAM Visualization</h5>
          <img
            src={prediction.gradcam_image}
            alt="Grad-CAM"
            style={{ width: "100%", borderRadius: 12 }}
          />
          <p className="text-muted mt-2 mb-0" style={{ fontSize: 12 }}>
            Highlighted regions indicate areas that most influenced the model’s decision.
          </p>
        </div>
      ) : wantGradcam ? (
        <p className="text-muted mt-3 mb-0" style={{ fontSize: 12 }}>
          Grad-CAM was requested, but the backend did not return an image.
        </p>
      ) : null}

      {showSecurity ? (
        <Alert variant="light" className="security-alert mt-3">
          <FaShieldAlt className="me-2" />
          Security: {securityFlags.encrypted ? "Encrypted" : "Not encrypted"} •{" "}
          {securityFlags.anonymized ? "Anonymized" : "Not anonymized"} • GDPR:{" "}
          {securityFlags.gdpr_compliant ? "Yes" : "No"} • PDPA:{" "}
          {securityFlags.pdpa_compliant ? "Yes" : "No"}
          {storedUpload !== null ? <> • Stored: {storedUpload ? "Yes" : "No"}</> : null}
        </Alert>
      ) : (
        <Alert variant="light" className="security-alert mt-3">
          <FaShieldAlt className="me-2" />
          Your data is anonymized and protected by security controls. GDPR/PDPA aligned.
        </Alert>
      )}

      {encryptedUploadId ? (
        <div style={{ fontSize: 12, opacity: 0.85, marginTop: 6 }}>
          Encrypted upload ID:{" "}
          <span style={{ fontFamily: "monospace" }}>{String(encryptedUploadId)}</span>
        </div>
      ) : null}
    </div>
  ) : (
    <div className="no-results">
      <div className="info-icon">
        <FaBrain />
      </div>
      <h3>Results</h3>
      <p className="text-muted">
        Upload an image and click "Analyze Now" to see the results here
      </p>
    </div>
  );

  return (
    <div className="app-container">
      <div className="app-header-top">
        <Container>
          <Row className="align-items-center">
            <Col>
              <h1 className="mb-0">
                Diabetic Retinopathy Prediction and Classification Web Application System
              </h1>
              <p className="mb-0 text-light">Powered by RA EfficientNetB3 Deep Learning</p>
              <div className="mt-2">{backendBadge}</div>

              <div className="mt-2" style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <Button variant="outline-light" size="sm" onClick={openModelInfo}>
                  <FaInfoCircle className="me-2" />
                  Model Info
                </Button>
                <Button variant="outline-light" size="sm" onClick={openPrivacyNotice}>
                  <FaShieldAlt className="me-2" />
                  Privacy Notice
                </Button>
              </div>
            </Col>
            <Col xs="auto">
              <div className="security-badge">
                <FaShieldAlt className="me-2" />
                GDPR/PDPA Aligned
              </div>
            </Col>
          </Row>
        </Container>
      </div>

      <Container className="mt-4">
        {error ? (
          <Alert
            variant="danger"
            className="alert-modern"
            dismissible
            onClose={() => setError("")}
          >
            <strong>Error:</strong> {error}
          </Alert>
        ) : null}

        <Row className="g-4">
          <Col lg={6}>
            <Card className="upload-card h-100">
              <Card.Body className="d-flex flex-column">
                <h3 className="mb-4">
                  <FaImage className="me-2" />
                  Upload Retinal Image
                </h3>

                {uploadSection}

                {loading ? (
                  <div className="loading-container">
                    <div className="loading-spinner-modern" />
                    <p className="mt-3">System is analyzing the retinal image...</p>
                  </div>
                ) : null}
              </Card.Body>
            </Card>
          </Col>

          <Col lg={6}>
            <Card className="results-card h-100">
              <Card.Body>{resultsContent}</Card.Body>
            </Card>
          </Col>
        </Row>

        <Alert variant="info" className="disclaimer-alert mt-4">
          <strong>Medical Disclaimer:</strong> This tool is for educational and
          screening purposes only. Always consult a qualified ophthalmologist for
          diagnosis and treatment.
        </Alert>
      </Container>

      <Modal show={showModelInfo} onHide={() => setShowModelInfo(false)} centered size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Model Information</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {modelInfoLoading ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Spinner animation="border" size="sm" />
              <span>Loading model info...</span>
            </div>
          ) : modelInfoError ? (
            <Alert variant="danger">
              <strong>Error:</strong> {modelInfoError}
            </Alert>
          ) : modelInfo ? (
            <div style={{ fontSize: 14 }}>
              <p className="mb-2">
                <strong>Name:</strong> {modelInfo.model_name || "N/A"}
              </p>
              <p className="mb-2">
                <strong>Input shape:</strong>{" "}
                {Array.isArray(modelInfo.input_shape)
                  ? modelInfo.input_shape.join(" × ")
                  : "N/A"}
              </p>
              <p className="mb-2">
                <strong>Classes:</strong>{" "}
                {modelInfo.classes
                  ? Object.values(modelInfo.classes).map(toSafeClassLabel).join(", ")
                  : "N/A"}
              </p>
              <p className="mb-2">
                <strong>Model loaded:</strong> {String(Boolean(modelInfo.model_loaded))}
              </p>
              {modelInfo.model_load_error ? (
                <Alert variant="warning" className="mt-3">
                  <strong>Load error:</strong> {String(modelInfo.model_load_error)}
                </Alert>
              ) : null}
              {modelInfo.security_features ? (
                <div className="mt-3">
                  <strong>Security features:</strong>
                  <ul className="mb-0">
                    {Object.entries(modelInfo.security_features).map(([k, v]) => (
                      <li key={k}>
                        {k}: {Array.isArray(v) ? v.join(", ") : String(v)}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="text-muted mb-0">No model info available.</p>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModelInfo(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal show={showPrivacy} onHide={() => setShowPrivacy(false)} centered size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Privacy Notice</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {privacyLoading ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Spinner animation="border" size="sm" />
              <span>Loading privacy notice...</span>
            </div>
          ) : privacyError ? (
            <Alert variant="danger">
              <strong>Error:</strong> {privacyError}
            </Alert>
          ) : privacyInfo ? (
            <div style={{ fontSize: 14 }}>
              {privacyInfo.controller ? (
                <>
                  <p className="mb-2">
                    <strong>Controller:</strong> {privacyInfo.controller.name || "N/A"}
                  </p>
                  <p className="mb-2">
                    <strong>Contact:</strong> {privacyInfo.controller.contact || "N/A"}
                  </p>
                  <p className="mb-2">
                    <strong>DPO:</strong> {privacyInfo.controller.dpo_contact || "N/A"}
                  </p>
                </>
              ) : null}

              {privacyInfo.processing_purposes ? (
                <>
                  <strong>Processing purposes:</strong>
                  <ul>
                    {privacyInfo.processing_purposes.map((x, i) => (
                      <li key={i}>{String(x)}</li>
                    ))}
                  </ul>
                </>
              ) : null}

              {privacyInfo.data_collected ? (
                <>
                  <strong>Data collected:</strong>
                  <ul>
                    {privacyInfo.data_collected.map((x, i) => (
                      <li key={i}>{String(x)}</li>
                    ))}
                  </ul>
                </>
              ) : null}

              {privacyInfo.security_measures ? (
                <>
                  <strong>Security measures:</strong>
                  <ul>
                    {privacyInfo.security_measures.map((x, i) => (
                      <li key={i}>{String(x)}</li>
                    ))}
                  </ul>
                </>
              ) : null}

              {privacyInfo.retention_period ? (
                <p className="mb-2">
                  <strong>Retention:</strong> {String(privacyInfo.retention_period)}
                </p>
              ) : null}

              {privacyInfo.rights ? (
                <>
                  <strong>Rights:</strong>
                  <ul className="mb-0">
                    {privacyInfo.rights.map((x, i) => (
                      <li key={i}>{String(x)}</li>
                    ))}
                  </ul>
                </>
              ) : null}
            </div>
          ) : (
            <p className="text-muted mb-0">No privacy notice available.</p>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowPrivacy(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}
