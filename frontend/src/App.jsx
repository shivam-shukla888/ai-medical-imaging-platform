import React, { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [activeTab, setActiveTab] = useState('inference')
  const [backendStatus, setBackendStatus] = useState({ online: false, device: 'unknown' })
  const [samples, setSamples] = useState([])
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [inputMode, setInputMode] = useState('sample') // 'sample' or 'upload'

  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [predictionResult, setPredictionResult] = useState(null)

  const [auditLogs, setAuditLogs] = useState([])
  const [monaiResult, setMonaiResult] = useState(null)
  const [monaiLoading, setMonaiLoading] = useState(false)

  // Check backend connectivity on mount
  useEffect(() => {
    checkBackend()
    fetchSamples()
  }, [])

  const checkBackend = async () => {
    try {
      const resp = await fetch(`${API_BASE}/ready`)
      if (resp.ok) {
        const data = await resp.json()
        setBackendStatus({ online: true, device: data.device || 'CPU' })
      } else {
        setBackendStatus({ online: false, device: 'offline' })
      }
    } catch {
      setBackendStatus({ online: false, device: 'offline' })
    }
  }

  const fetchSamples = async () => {
    try {
      const resp = await fetch(`${API_BASE}/samples`)
      if (resp.ok) {
        const data = await resp.json()
        setSamples(data.samples || [])
      }
    } catch (e) {
      console.warn('Could not load samples:', e)
    }
  }

  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (file.size > 5 * 1024 * 1024) {
      setErrorMsg('Uploaded file exceeds maximum allowed limit of 5 MB.')
      setUploadedFile(null)
      return
    }

    setErrorMsg(null)
    setUploadedFile(file)
  }

  const handleAnalyze = async () => {
    setLoading(true)
    setErrorMsg(null)
    setPredictionResult(null)

    try {
      let resp
      if (inputMode === 'sample') {
        resp = await fetch(`${API_BASE}/predict/sample/${selectedSampleIdx}?explain=true`, {
          method: 'POST',
        })
      } else {
        if (!uploadedFile) {
          setErrorMsg('Please select an image file to upload.')
          setLoading(false)
          return
        }
        const formData = new FormData()
        formData.append('file', uploadedFile)
        resp = await fetch(`${API_BASE}/predict?explain=true`, {
          method: 'POST',
          body: formData,
        })
      }

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.detail || `Server returned error ${resp.status}`)
      }

      const data = await resp.json()
      setPredictionResult(data)
    } catch (err) {
      setErrorMsg(err.message || 'Failed to communicate with prediction service.')
    } finally {
      setLoading(false)
    }
  }

  const fetchAuditLogs = async () => {
    try {
      const resp = await fetch(`${API_BASE}/audit-logs?limit=25`)
      if (resp.ok) {
        const data = await resp.json()
        setAuditLogs(data.events || [])
      }
    } catch (e) {
      console.error(e)
    }
  }

  const handleRunMonaiDemo = async () => {
    setMonaiLoading(true)
    try {
      const resp = await fetch(`${API_BASE}/monai/demo-3d`)
      if (resp.ok) {
        const data = await resp.json()
        setMonaiResult(data)
      } else {
        const err = await resp.json()
        alert(err.detail || 'MONAI demo failed.')
      }
    } catch (e) {
      alert('Error: ' + e.message)
    } finally {
      setMonaiLoading(false)
    }
  }

  const currentSample = samples.find((s) => s.index === Number(selectedSampleIdx))

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="header-banner">
        <div className="brand-title">
          <span style={{ fontSize: '2.5rem' }}>🩻</span>
          <div>
            <h1>AI Chest X-Ray Research Platform</h1>
            <p>Deep Learning Interpretability & Diagnostic Exploration on PneumoniaMNIST</p>
          </div>
        </div>
        <div>
          <span className={`status-badge ${backendStatus.online ? 'online' : 'offline'}`}>
            ● Backend: {backendStatus.online ? `Online (${backendStatus.device.toUpperCase()})` : 'Offline'}
          </span>
        </div>
      </header>

      {/* Clinical Disclaimer Banner */}
      <div className="disclaimer-banner">
        <strong>⚠️ Mandatory Clinical Research Disclaimer:</strong> Educational research demo only.
        Not intended for clinical use, patient diagnosis, treatment planning, triage, or medical decision-making.
      </div>

      {/* Navigation Tabs */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'inference' ? 'active' : ''}`}
          onClick={() => setActiveTab('inference')}
        >
          🔬 Diagnostic Inference & Grad-CAM
        </button>
        <button
          className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          📊 Benchmark Evaluation Metrics
        </button>
        <button
          className={`tab-btn ${activeTab === 'monai' ? 'active' : ''}`}
          onClick={() => setActiveTab('monai')}
        >
          🗂️ MONAI 3D Volumetric Demo
        </button>
        <button
          className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('audit')
            fetchAuditLogs()
          }}
        >
          📜 Platform Audit Log
        </button>
      </nav>

      {/* TAB 1: Inference & Grad-CAM */}
      {activeTab === 'inference' && (
        <div>
          <div className="card">
            <h2>Select Input Radiograph</h2>
            <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="inputMode"
                  value="sample"
                  checked={inputMode === 'sample'}
                  onChange={() => setInputMode('sample')}
                />
                Benchmark Test Sample
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                <input
                  type="radio"
                  name="inputMode"
                  value="upload"
                  checked={inputMode === 'upload'}
                  onChange={() => setInputMode('upload')}
                />
                Custom Radiograph Upload
              </label>
            </div>

            {inputMode === 'sample' ? (
              <div className="grid-2" style={{ alignItems: 'center' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '6px', fontWeight: 600 }}>
                    Choose Held-Out Test Image:
                  </label>
                  <select
                    className="select-input"
                    value={selectedSampleIdx}
                    onChange={(e) => setSelectedSampleIdx(Number(e.target.value))}
                  >
                    {samples.map((s) => (
                      <option key={s.index} value={s.index}>
                        Sample #{s.index} — Ground Truth: {s.true_label_name}
                      </option>
                    ))}
                  </select>
                </div>
                {currentSample && (
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '6px' }}>
                      Reference Label: <strong>{currentSample.true_label_name}</strong>
                    </p>
                    <img
                      src={currentSample.preview_base64}
                      alt="Thumbnail"
                      style={{ width: '80px', height: '80px', border: '1px solid #cbd5e1', borderRadius: '4px' }}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div>
                <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '6px', fontWeight: 600 }}>
                  Upload Image (PNG, JPG, max 5 MB):
                </label>
                <input type="file" className="file-input" accept="image/png,image/jpeg" onChange={handleFileUpload} />
              </div>
            )}

            {errorMsg && <div className="alert-error">⚠️ {errorMsg}</div>}

            <button
              className="btn-primary"
              onClick={handleAnalyze}
              disabled={loading || !backendStatus.online}
              style={{ marginTop: '12px' }}
            >
              {loading ? 'Analyzing with Grad-CAM...' : '🚀 Run Prediction & Grad-CAM'}
            </button>
          </div>

          {/* Results Display */}
          {predictionResult && (
            <div className="card">
              <h2>Diagnostic Prediction & Explainability</h2>
              <p style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '20px' }}>
                Inference Latency: <strong>{predictionResult.inference_time_ms} ms</strong> | Architecture:{' '}
                <strong>{predictionResult.model_metadata?.architecture || 'ChestXRayMiniCNN'}</strong>
              </p>

              <div className="grid-3" style={{ marginBottom: '24px' }}>
                <div className="image-preview-container">
                  <img src={predictionResult.image_preview_base64} alt="Input Radiograph" />
                  <span className="image-caption">1. Input Grayscale (28x28)</span>
                </div>

                <div className="image-preview-container">
                  {predictionResult.explanation?.heatmap_base64 ? (
                    <img src={predictionResult.explanation.heatmap_base64} alt="Grad-CAM Heatmap" />
                  ) : (
                    <div style={{ padding: '40px 0', color: '#94a3b8' }}>Heatmap Unavailable</div>
                  )}
                  <span className="image-caption">2. Grad-CAM Activation</span>
                </div>

                <div className="image-preview-container">
                  {predictionResult.explanation?.overlay_base64 ? (
                    <img src={predictionResult.explanation.overlay_base64} alt="Grad-CAM Overlay" />
                  ) : (
                    <div style={{ padding: '40px 0', color: '#94a3b8' }}>Overlay Unavailable</div>
                  )}
                  <span className="image-caption">3. Anatomical Overlay</span>
                </div>
              </div>

              {/* Classification Output */}
              <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                  <span style={{ fontSize: '1rem', fontWeight: 600 }}>Classification:</span>
                  <span
                    className={
                      predictionResult.prediction.class_name === 'Pneumonia'
                        ? 'badge-pneumonia'
                        : 'badge-normal'
                    }
                  >
                    {predictionResult.prediction.class_name === 'Pneumonia' ? '🦠 Pneumonia' : '🫁 Normal'}
                  </span>
                </div>

                <h3 style={{ fontSize: '1rem', marginBottom: '8px' }}>Model Scores</h3>
                <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '12px' }}>
                  {predictionResult.prediction.score_note}
                </p>

                {Object.entries(predictionResult.prediction.model_scores || {}).map(([cname, score]) => (
                  <div key={cname} className="score-row">
                    <span className="score-label">{cname}</span>
                    <div className="score-bar-bg">
                      <div className="score-bar-fill" style={{ width: `${score * 100}%` }} />
                    </div>
                    <span className="score-percent">{(score * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Metrics */}
      {activeTab === 'metrics' && (
        <div>
          <div className="card">
            <h2>Official PneumoniaMNIST Test Split Evaluation (624 Samples)</h2>
            <div className="grid-4" style={{ marginBottom: '20px' }}>
              <div className="metric-box">
                <div className="metric-value">89.7%</div>
                <div className="metric-label">Accuracy</div>
              </div>
              <div className="metric-box">
                <div className="metric-value">98.2%</div>
                <div className="metric-label">Sensitivity (Recall)</div>
              </div>
              <div className="metric-box">
                <div className="metric-value">75.6%</div>
                <div className="metric-label">Specificity</div>
              </div>
              <div className="metric-box">
                <div className="metric-value">0.942</div>
                <div className="metric-label">ROC-AUC</div>
              </div>
            </div>

            <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <h3 style={{ fontSize: '1rem', marginBottom: '8px' }}>Confusion Matrix (Test Split)</h3>
              <p style={{ fontSize: '0.9rem', color: '#475569' }}>
                • <strong>True Positives (Pneumonia detected)</strong>: 383 / 390 (98.2%)<br />
                • <strong>True Negatives (Normal detected)</strong>: 177 / 234 (75.6%)<br />
                • <strong>False Positives (Normal flagged as Pneumonia)</strong>: 57<br />
                • <strong>False Negatives (Pneumonia missed)</strong>: 7
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: MONAI 3D Demo */}
      {activeTab === 'monai' && (
        <div className="card">
          <h2>MONAI 3D Volumetric U-Net Segmentation Demo</h2>
          <p style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '16px' }}>
            Educational demonstration executing a 3D U-Net forward segmentation pass on a simulated 3D volumetric phantom without large dataset downloads.
          </p>

          <button
            className="btn-primary"
            onClick={handleRunMonaiDemo}
            disabled={monaiLoading}
            style={{ width: 'auto', marginBottom: '20px' }}
          >
            {monaiLoading ? 'Running 3D U-Net...' : '▶ Run MONAI 3D Volumetric Demo'}
          </button>

          {monaiResult && (
            <div style={{ background: '#0f172a', color: '#e2e8f0', padding: '16px', borderRadius: '8px', fontFamily: 'monospace' }}>
              <pre>{JSON.stringify(monaiResult, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: Audit Logs */}
      {activeTab === 'audit' && (
        <div className="card">
          <h2>Platform Research Audit Events (Append-Only)</h2>
          <p style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '16px' }}>
            Demonstrates research platform auditability. Records timestamps, prediction outcomes, and latencies without capturing raw patient images or identifiers.
          </p>

          {auditLogs.length === 0 ? (
            <p style={{ color: '#94a3b8' }}>No audit events logged yet. Execute an inference request to view events.</p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ background: '#f1f5f9', textAlign: 'left' }}>
                    <th style={{ padding: '8px' }}>Timestamp</th>
                    <th style={{ padding: '8px' }}>Event</th>
                    <th style={{ padding: '8px' }}>Status</th>
                    <th style={{ padding: '8px' }}>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {auditLogs.map((log, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px', color: '#64748b' }}>{log.timestamp}</td>
                      <td style={{ padding: '8px', fontWeight: 600 }}>{log.event_type}</td>
                      <td style={{ padding: '8px' }}>
                        <span style={{ color: log.status === 'SUCCESS' ? '#16a34a' : '#dc2626', fontWeight: 600 }}>
                          {log.status}
                        </span>
                      </td>
                      <td style={{ padding: '8px', color: '#475569' }}>{JSON.stringify(log.details)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
