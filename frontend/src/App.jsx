import { useState, useEffect, useRef } from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

import "./App.css";

const API_BASE_URL = "http://127.0.0.1:7860";

function App() {
  const [activePage, setActivePage] =
    useState("dashboard");

  // ========================================
  // BACKEND STATUS
  // ========================================

  const [backendStatus, setBackendStatus] =
    useState("checking");

  // ========================================
  // GLOBAL / DASHBOARD LOADING & ERROR
  // ========================================

  const [dashboardLoading, setDashboardLoading] =
    useState(true);

  const [dashboardError, setDashboardError] =
    useState("");

  // ========================================
  // MODEL CONFIGURATION
  // ========================================

  const [modelConfig, setModelConfig] =
    useState({
      endpoint: "",
      apiKey: "",
      modelName: "",
      datasetType: "default",
      testMode: true,
    });

  // ========================================
  // DATASET STATES
  // ========================================

  const [uploadedDataset, setUploadedDataset] =
    useState(null);

  const [uploadStatus, setUploadStatus] =
    useState("");

  const [uploadError, setUploadError] =
    useState("");

  const [isUploading, setIsUploading] =
    useState(false);

  // ========================================
  // EVALUATION STATES
  // ========================================

  const [evaluationStatus, setEvaluationStatus] =
    useState("");

  const [evaluationData, setEvaluationData] =
    useState(null);

  const [evaluationHistory, setEvaluationHistory] =
    useState([]);

  const [historyLoading, setHistoryLoading] =
    useState(false);

  const [historyError, setHistoryError] =
    useState("");

  // ========================================
  // LIVE PROGRESS STATES
  // ========================================

  const [isEvaluating, setIsEvaluating] =
    useState(false);

  const [evaluationId, setEvaluationId] =
    useState(null);

  const [progressCurrent, setProgressCurrent] =
    useState(0);

  const [progressTotal, setProgressTotal] =
    useState(0);

  const [progressCategory, setProgressCategory] =
    useState("");

  const [progressTest, setProgressTest] =
    useState("");

  const pollingRef = useRef(null);

  // ========================================
  // CHART COLORS
  // ========================================

  const chartColors = [
    "#8b6fcb",
    "#a98be8",
    "#c5b3f5",
    "#7055a3",
    "#d8ccf5",
    "#9d85c7",
  ];

  // ========================================
  // SAFE RESPONSE PARSER
  // ========================================

  const getResponseData = async (response) => {
    const contentType =
      response.headers.get("content-type");

    if (
      contentType &&
      contentType.includes("application/json")
    ) {
      return await response.json();
    }

    const text = await response.text();

    return {
      detail:
        text ||
        "The server returned an unexpected response.",
    };
  };

  // ========================================
  // CHECK BACKEND STATUS
  // ========================================

  const checkBackendStatus = async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/health`
      );

      if (!response.ok) {
        throw new Error(
          "Backend is unavailable."
        );
      }

      setBackendStatus("connected");

    } catch (error) {
      console.error(
        "Backend connection error:",
        error
      );

      setBackendStatus("disconnected");
    }
  };

  // ========================================
  // FETCH EVALUATION HISTORY
  // ========================================

  const fetchHistory = async (
    showDashboardLoading = false
  ) => {
    setHistoryLoading(true);
    setHistoryError("");

    if (showDashboardLoading) {
      setDashboardLoading(true);
      setDashboardError("");
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/history`
      );

      const data =
        await getResponseData(response);

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Unable to fetch evaluation history."
        );
      }

      const history = data.history || [];

      setEvaluationHistory(history);

      if (history.length > 0) {
        setEvaluationData(history[0]);
      } else {
        setEvaluationData(null);
      }

    } catch (error) {
      console.error(
        "History error:",
        error
      );

      setHistoryError(
        `Unable to load evaluation history: ${error.message}`
      );

      if (showDashboardLoading) {
        setDashboardError(
          `Unable to load dashboard data: ${error.message}`
        );
      }

    } finally {
      setHistoryLoading(false);

      if (showDashboardLoading) {
        setDashboardLoading(false);
      }
    }
  };

  // ========================================
  // FETCH LATEST EVALUATION
  // ========================================

  const fetchLatestEvaluation = async () => {
    setDashboardError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/history/latest`
      );

      const data =
        await getResponseData(response);

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Unable to fetch latest evaluation."
        );
      }

      if (data.evaluation) {
        setEvaluationData(
          data.evaluation
        );
      }

    } catch (error) {
      console.error(
        "Latest evaluation error:",
        error
      );

      setDashboardError(
        `Unable to load the latest evaluation: ${error.message}`
      );
    }
  };

  // ========================================
  // STOP PROGRESS POLLING
  // ========================================

  const stopProgressPolling = () => {
    if (pollingRef.current) {
      clearInterval(
        pollingRef.current
      );

      pollingRef.current = null;
    }
  };

  // ========================================
  // RETRY DASHBOARD
  // ========================================

  const retryDashboard = async () => {
    await checkBackendStatus();
    await fetchHistory(true);
  };

  // ========================================
  // LOAD DATA ON STARTUP
  // ========================================

  useEffect(() => {
    checkBackendStatus();

    fetchHistory(true);

    const backendCheckInterval =
      setInterval(
        checkBackendStatus,
        5000
      );

    return () => {
      stopProgressPolling();

      clearInterval(
        backendCheckInterval
      );
    };
  }, []);

  // ========================================
  // UPLOAD CUSTOM DATASET
  // ========================================

  const handleDatasetUpload =
    async (event) => {

      const file =
        event.target.files[0];

      if (!file) {
        return;
      }

      setUploadError("");
      setUploadStatus("");

      if (
        !file.name
          .toLowerCase()
          .endsWith(".csv")
      ) {
        setUploadedDataset(null);

        setUploadError(
          "Please select a valid CSV file."
        );

        return;
      }

      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      setIsUploading(true);

      setUploadStatus(
        "Uploading dataset..."
      );

      try {
        const response =
          await fetch(
            `${API_BASE_URL}/upload-dataset`,
            {
              method: "POST",
              body: formData,
            }
          );

        const data =
          await getResponseData(response);

        if (!response.ok) {
          throw new Error(
            data.detail ||
            "Dataset upload failed."
          );
        }

        setUploadedDataset(data);

        setUploadStatus(
          `Dataset uploaded successfully! ${data.total_tests} test cases found.`
        );

      } catch (error) {
        console.error(
          "Upload error:",
          error
        );

        setUploadedDataset(null);

        setUploadStatus("");

        setUploadError(
          `Upload failed: ${error.message}`
        );

      } finally {
        setIsUploading(false);
      }
    };

  // ========================================
  // CHECK EVALUATION PROGRESS
  // ========================================

  const checkEvaluationProgress =
    async (currentEvaluationId) => {

      if (!currentEvaluationId) {
        return;
      }

      try {
        const response =
          await fetch(
            `${API_BASE_URL}/evaluation-progress/${currentEvaluationId}`
          );

        const data =
          await getResponseData(response);

        if (!response.ok) {
          throw new Error(
            data.detail ||
            "Unable to fetch evaluation progress."
          );
        }

        setProgressCurrent(
          data.completed || 0
        );

        setProgressTotal(
          data.total || 0
        );

        setProgressCategory(
          data.current_category || ""
        );

        setProgressTest(
          data.current_test || ""
        );

        if (
          data.status === "completed"
        ) {
          stopProgressPolling();

          setProgressCurrent(
            data.completed ||
            data.total ||
            0
          );

          setProgressTotal(
            data.total ||
            data.completed ||
            0
          );

          setProgressCategory("");
          setProgressTest("");

          setIsEvaluating(false);
          setEvaluationId(null);

          setEvaluationStatus(
            `Evaluation completed successfully! ${
              data.total ||
              data.completed ||
              0
            } test cases evaluated.`
          );

          await fetchLatestEvaluation();

          await fetchHistory();

          setActivePage("dashboard");
        }

        if (
          data.status === "failed"
        ) {
          stopProgressPolling();

          setIsEvaluating(false);
          setEvaluationId(null);

          setEvaluationStatus(
            `Evaluation failed: ${
              data.error ||
              "Unknown error"
            }`
          );
        }

      } catch (error) {
        console.error(
          "Progress error:",
          error
        );

        stopProgressPolling();

        setIsEvaluating(false);
        setEvaluationId(null);

        setEvaluationStatus(
          `Unable to track evaluation progress: ${error.message}`
        );
      }
    };

  // ========================================
  // START EVALUATION
  // ========================================

  const handleStartEvaluation =
    async () => {

      setEvaluationStatus("");

      if (
        backendStatus !== "connected"
      ) {
        setEvaluationStatus(
          "Backend is not connected. Please make sure the SentinelLLM backend is running."
        );

        return;
      }

      if (
        !modelConfig.endpoint.trim() ||
        !modelConfig.apiKey.trim() ||
        !modelConfig.modelName.trim()
      ) {
        setEvaluationStatus(
          "Please enter the API endpoint, API key, and model name."
        );

        return;
      }

      if (
        modelConfig.datasetType ===
          "custom" &&
        !uploadedDataset
      ) {
        setEvaluationStatus(
          "Please upload a custom dataset first."
        );

        return;
      }

      stopProgressPolling();

      setProgressCurrent(0);
      setProgressTotal(0);
      setProgressCategory("");
      setProgressTest("");
      setEvaluationId(null);

      setIsEvaluating(true);

      setEvaluationStatus(
        "Starting evaluation..."
      );

      const requestBody = {
        endpoint:
          modelConfig.endpoint.trim(),

        api_key:
          modelConfig.apiKey.trim(),

        model_name:
          modelConfig.modelName.trim(),

        dataset_type:
          modelConfig.datasetType,

        filename:
          modelConfig.datasetType ===
            "custom"
            ? uploadedDataset.filename
            : null,

        test_mode:
          modelConfig.testMode,
      };

      try {
        const response =
          await fetch(
            `${API_BASE_URL}/evaluate`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify(
                  requestBody
                ),
            }
          );

        const data =
          await getResponseData(response);

        if (!response.ok) {
          throw new Error(
            data.detail ||
            "Evaluation failed to start."
          );
        }

        if (!data.evaluation_id) {
          throw new Error(
            "Backend did not return an evaluation ID."
          );
        }

        const newEvaluationId =
          data.evaluation_id;

        setEvaluationId(
          newEvaluationId
        );

        setEvaluationStatus(
          "Evaluation is running..."
        );

        checkEvaluationProgress(
          newEvaluationId
        );

        pollingRef.current =
          setInterval(
            () => {
              checkEvaluationProgress(
                newEvaluationId
              );
            },
            1500
          );

      } catch (error) {

        console.error(
          "Evaluation start error:",
          error
        );

        stopProgressPolling();

        setIsEvaluating(false);
        setEvaluationId(null);

        setEvaluationStatus(
          `Evaluation failed to start: ${error.message}`
        );
      }
    };

  // ========================================
  // PREPARE CATEGORY DATA
  // ========================================

  const categoryData =
    evaluationData
      ? Object.entries(
          evaluationData.category_scores || {}
        ).map(
          ([name, score]) => ({
            name:
              name
                .replace(/_/g, " ")
                .replace(
                  /\b\w/g,
                  (letter) =>
                    letter.toUpperCase()
                ),

            score:
              Number(score) || 0,
          })
        )
      : [];

  // ========================================
  // NORMALIZE RECOMMENDATION
  // FIXES [object Object]
  // ========================================

  const normalizeRecommendation =
    (recommendation) => {

      if (
        typeof recommendation ===
        "string"
      ) {
        return recommendation;
      }

      if (
        recommendation &&
        typeof recommendation ===
          "object"
      ) {

        return (
          recommendation.recommendation ||
          recommendation.message ||
          recommendation.text ||
          recommendation.description ||
          recommendation.content ||
          recommendation.suggestion ||
          recommendation.action ||
          JSON.stringify(recommendation)
        );
      }

      return String(
        recommendation || ""
      );
    };

  // ========================================
  // GENERATE RECOMMENDATIONS
  // ========================================

  const getRecommendations = () => {

    if (!evaluationData) {
      return [];
    }

    // USE BACKEND RECOMMENDATIONS
    // AND CONVERT OBJECTS TO TEXT

    if (
      Array.isArray(
        evaluationData.recommendations
      ) &&
      evaluationData.recommendations.length > 0
    ) {

      return evaluationData.recommendations
        .map(
          normalizeRecommendation
        )
        .filter(
          (recommendation) =>
            recommendation &&
            recommendation.trim() !== ""
        );
    }

    // FALLBACK RECOMMENDATIONS

    const recommendations = [];

    const categoryScores =
      evaluationData.category_scores || {};

    const sortedCategories =
      Object.entries(
        categoryScores
      ).sort(
        (a, b) =>
          Number(a[1]) -
          Number(b[1])
      );

    sortedCategories.forEach(
      ([category, score]) => {

        const numericScore =
          Number(score) || 0;

        if (numericScore >= 80) {
          return;
        }

        if (
          category === "hallucination"
        ) {
          recommendations.push(
            numericScore < 50
              ? "High priority: Strengthen factual grounding and response verification. The model should avoid presenting uncertain or unsupported information as factual."
              : "Improve factual grounding and verification mechanisms to further reduce hallucinated or unsupported information."
          );
        }

        else if (
          category === "bias"
        ) {
          recommendations.push(
            numericScore < 50
              ? "High priority: Review responses for unfair assumptions or unequal treatment across user groups and strengthen fairness testing."
              : "Improve fairness evaluation and review responses for potential bias across different user groups and scenarios."
          );
        }

        else if (
          category === "toxicity"
        ) {
          recommendations.push(
            numericScore < 50
              ? "High priority: Strengthen safety filters and harmful-content detection to reduce toxic or inappropriate responses."
              : "Improve safety filters and response policies to further reduce toxic, harmful, or inappropriate outputs."
          );
        }

        else if (
          category === "jailbreak"
        ) {
          recommendations.push(
            numericScore < 50
              ? "High priority: Strengthen jailbreak resistance by enforcing instruction hierarchy and consistently refusing unsafe attempts to bypass safeguards."
              : "Improve jailbreak resistance by strengthening instruction hierarchy handling and refusal behavior for unsafe requests."
          );
        }

        else if (
          category === "prompt_injection"
        ) {
          recommendations.push(
            numericScore < 50
              ? "High priority: Strengthen prompt-injection defenses so untrusted instructions cannot override system-level rules or intended behavior."
              : "Improve prompt-injection protection by ensuring untrusted instructions cannot override system-level behavior."
          );
        }

        else if (
          category === "reasoning"
        ) {
          recommendations.push(
            numericScore < 50
              ? "High priority: Improve multi-step reasoning reliability and validate answers more carefully on logical and complex problem-solving tasks."
              : "Improve reasoning reliability by testing the model with more complex logical, mathematical, and multi-step tasks."
          );
        }
      }
    );

    const overallScore =
      Number(
        evaluationData.overall_score
      ) || 0;

    if (overallScore < 50) {
      recommendations.push(
        "Overall performance requires significant improvement. Focus first on the lowest-scoring categories and repeat the evaluation after applying safety improvements."
      );
    }

    else if (overallScore < 80) {
      recommendations.push(
        "Prioritize the lowest-performing categories and run another evaluation after applying targeted improvements."
      );
    }

    if (
      recommendations.length === 0
    ) {
      recommendations.push(
        "The model performed strongly across all evaluated categories. Continue testing with broader benchmarks and more challenging edge cases to maintain reliability."
      );
    }

    return recommendations;
  };

  const recommendations =
    getRecommendations();

  // ========================================
  // PROGRESS PERCENTAGE
  // ========================================

  const progressPercentage =
    progressTotal > 0
      ? Math.min(
          100,
          Math.round(
            (
              progressCurrent /
              progressTotal
            ) * 100
          )
        )
      : 0;

  // ========================================
  // FORMAT CATEGORY NAME
  // ========================================

  const formatCategoryName =
    (category) => {

      if (!category) {
        return "";
      }

      return category
        .replace(/_/g, " ")
        .replace(
          /\b\w/g,
          (letter) =>
            letter.toUpperCase()
        );
    };

  // ========================================
  // DASHBOARD COMPONENT
  // ========================================

  const Dashboard = () => (
    <>
      {dashboardLoading ? (

        <section className="content-card loading-card">

          <div className="loading-spinner">
          </div>

          <h2>
            Loading Dashboard
          </h2>

          <p>
            Fetching your latest evaluation data...
          </p>

        </section>

      ) : dashboardError ? (

        <section className="content-card error-card">

          <h2>
            Unable to Load Dashboard
          </h2>

          <p>
            {dashboardError}
          </p>

          <button
            className="evaluate-button"
            onClick={retryDashboard}
          >
            ↻ Retry
          </button>

        </section>

      ) : !evaluationData ? (

        <section className="content-card">

          <h2>
            No Evaluation Yet
          </h2>

          <p>
            Connect an LLM and run an evaluation
            to see real safety scores here.
          </p>

          <button
            className="evaluate-button"
            onClick={() =>
              setActivePage("evaluate")
            }
          >
            🔍 Evaluate an LLM
          </button>

        </section>

      ) : (

        <>

          <section className="score-grid">

            <div className="stat-card overall-card">

              <p>
                Overall SentinelLLM Score
              </p>

              <h2>
                {evaluationData.overall_score}
                <span>/100</span>
              </h2>

              <small>
                Latest evaluation result
              </small>

            </div>

            <div className="stat-card">

              <p>
                Total Test Cases
              </p>

              <h2>
                {evaluationData.total_tests}
              </h2>

              <small>
                Evaluated test prompts
              </small>

            </div>

            <div className="stat-card">

              <p>
                Passed Tests
              </p>

              <h2>
                {evaluationData.passed_tests}
              </h2>

              <small>
                Responses meeting evaluation criteria
              </small>

            </div>

            <div className="stat-card">

              <p>
                Needs Attention
              </p>

              <h2>
                {evaluationData.needs_attention}
              </h2>

              <small>
                Tests requiring attention
              </small>

            </div>

          </section>

          <section className="dashboard-section">

            <div className="section-heading">

              <h2>
                Category Performance
              </h2>

              <p>
                Scores from the latest evaluation.
              </p>

            </div>

            <div className="category-grid">

              {categoryData.map(
                (category) => (

                  <div
                    className="category-card"
                    key={category.name}
                  >

                    <div className="category-top">

                      <h3>
                        {category.name}
                      </h3>

                      <span>
                        {category.score}/100
                      </span>

                    </div>

                    <div className="progress-bar">

                      <div
                        className="progress-fill"
                        style={{
                          width:
                            `${Math.min(
                              100,
                              category.score
                            )}%`,
                        }}
                      >
                      </div>

                    </div>

                  </div>
                )
              )}

            </div>

          </section>

          {categoryData.length > 0 && (

            <section className="chart-section">

              <div className="chart-heading">

                <h2>
                  Evaluation Score Distribution
                </h2>

                <p>
                  Performance across evaluation categories.
                </p>

              </div>

              <div className="chart-container">

                <ResponsiveContainer
                  width="100%"
                  height={400}
                >

                  <PieChart>

                    <Pie
                      data={categoryData}
                      dataKey="score"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={80}
                      outerRadius={130}
                      paddingAngle={3}
                    >

                      {categoryData.map(
                        (
                          category,
                          index
                        ) => (

                          <Cell
                            key={`cell-${index}`}
                            fill={
                              chartColors[
                                index %
                                chartColors.length
                              ]
                            }
                          />

                        )
                      )}

                    </Pie>

                    <Tooltip />

                    <Legend />

                  </PieChart>

                </ResponsiveContainer>

              </div>

            </section>

          )}

          <section
            className="recommendations-section"
          >

            <div
              className="recommendations-heading"
            >

              <h2>
                AI Safety Recommendations
              </h2>

              <p>
                Suggested improvements based on
                the latest evaluation results.
              </p>

            </div>

            <div
              className="recommendations-list"
            >

              {recommendations.map(
                (
                  recommendation,
                  index
                ) => (

                  <div
                    className="recommendation-item"
                    key={index}
                  >

                    <div
                      className="recommendation-number"
                    >
                      {index + 1}
                    </div>

                    <div
                      className="recommendation-content"
                    >
                      <p>
                        {recommendation}
                      </p>
                    </div>

                  </div>

                )
              )}

            </div>

          </section>

        </>

      )}
    </>
  );

  // ========================================
  // APP UI
  // ========================================

  return (
    <div className="app">

      <aside className="sidebar">

        <div className="logo">

          <div className="logo-icon">
            🛡️
          </div>

          <div>

            <h2>
              SentinelLLM
            </h2>

            <span>
              LLM Safety Evaluator
            </span>

          </div>

        </div>

        <nav>

          <button
            className={
              activePage === "dashboard"
                ? "active"
                : ""
            }
            onClick={() =>
              setActivePage("dashboard")
            }
          >
            📊 Dashboard
          </button>

          <button
            className={
              activePage === "evaluate"
                ? "active"
                : ""
            }
            onClick={() =>
              setActivePage("evaluate")
            }
          >
            🔍 Evaluate Model
          </button>

          <button
            className={
              activePage === "upload"
                ? "active"
                : ""
            }
            onClick={() =>
              setActivePage("upload")
            }
          >
            📁 Upload Dataset
          </button>

          <button
            className={
              activePage === "history"
                ? "active"
                : ""
            }
            onClick={() => {
              setActivePage("history");
              fetchHistory();
            }}
          >
            🕘 Evaluation History
          </button>

        </nav>

        <div className="sidebar-bottom">

          <p>
            🛡️ AI Safety First
          </p>

          <small>
            SentinelLLM v1.0
          </small>

        </div>

      </aside>

      <main className="main-content">

        <header>

          <div>

            <h1>

              {activePage === "dashboard" &&
                "Safety Dashboard"}

              {activePage === "evaluate" &&
                "Evaluate an LLM"}

              {activePage === "upload" &&
                "Upload Custom Dataset"}

              {activePage === "history" &&
                "Evaluation History"}

            </h1>

            <p>
              Evaluate and analyze Large Language Model
              safety, reliability, and robustness.
            </p>

          </div>

          <div
            className={
              `status ${backendStatus}`
            }
          >

            <span className="status-dot">
            </span>

            {backendStatus ===
              "connected" &&
              "Backend Connected"}

            {backendStatus ===
              "checking" &&
              "Checking Backend..."}

            {backendStatus ===
              "disconnected" &&
              "Backend Disconnected"}

          </div>

        </header>

        {/* DASHBOARD PAGE */}

        {activePage === "dashboard" && (
          <Dashboard />
        )}

        {/* EVALUATE MODEL PAGE */}

        {activePage === "evaluate" && (

          <section className="content-card evaluation-form">

            <div className="form-heading">

              <h2>
                Connect an LLM
              </h2>

              <p>
                Configure the LLM you want
                SentinelLLM to evaluate.
              </p>

            </div>

            <div className="evaluation-fields">

              <div className="form-group">

                <label>
                  API Endpoint
                </label>

                <input
                  type="text"
                  placeholder="Enter the LLM API endpoint"
                  value={
                    modelConfig.endpoint
                  }
                  disabled={isEvaluating}
                  onChange={(event) =>
                    setModelConfig({
                      ...modelConfig,
                      endpoint:
                        event.target.value,
                    })
                  }
                />

              </div>

              <div className="form-group">

                <label>
                  API Key
                </label>

                <input
                  type="password"
                  placeholder="Enter your API key"
                  value={
                    modelConfig.apiKey
                  }
                  disabled={isEvaluating}
                  onChange={(event) =>
                    setModelConfig({
                      ...modelConfig,
                      apiKey:
                        event.target.value,
                    })
                  }
                />

              </div>

              <div className="form-group">

                <label>
                  Model Name
                </label>

                <input
                  type="text"
                  placeholder="Example: gemini-2.5-flash"
                  value={
                    modelConfig.modelName
                  }
                  disabled={isEvaluating}
                  onChange={(event) =>
                    setModelConfig({
                      ...modelConfig,
                      modelName:
                        event.target.value,
                    })
                  }
                />

              </div>

            </div>

            {/* TEST MODE */}

            <div className="dataset-section">

              <h3>
                Evaluation Mode
              </h3>

              <label className="radio-option">

                <input
                  type="checkbox"
                  checked={
                    modelConfig.testMode
                  }
                  disabled={isEvaluating}
                  onChange={(event) =>
                    setModelConfig({
                      ...modelConfig,
                      testMode:
                        event.target.checked,
                    })
                  }
                />

                <div className="dataset-text">

                  <span>
                    Test Mode
                  </span>

                  <small>
                    Run only 10 prompts across
                    multiple safety categories.
                  </small>

                </div>

              </label>

            </div>

            {/* DATASET */}

            <div className="dataset-section">

              <h3>
                Evaluation Dataset
              </h3>

              <div className="dataset-options">

                <label className="radio-option">

                  <input
                    type="radio"
                    name="dataset"
                    value="default"
                    disabled={isEvaluating}
                    checked={
                      modelConfig.datasetType ===
                      "default"
                    }
                    onChange={(event) =>
                      setModelConfig({
                        ...modelConfig,
                        datasetType:
                          event.target.value,
                      })
                    }
                  />

                  <div className="dataset-text">

                    <span>
                      Use SentinelLLM Default Benchmark
                    </span>

                    <small>

                      {modelConfig.testMode
                        ? "10 selected test prompts"
                        : "120 test prompts"}

                    </small>

                  </div>

                </label>

                <label className="radio-option">

                  <input
                    type="radio"
                    name="dataset"
                    value="custom"
                    disabled={isEvaluating}
                    checked={
                      modelConfig.datasetType ===
                      "custom"
                    }
                    onChange={(event) =>
                      setModelConfig({
                        ...modelConfig,
                        datasetType:
                          event.target.value,
                      })
                    }
                  />

                  <div className="dataset-text">

                    <span>
                      Use Uploaded Custom Dataset
                    </span>

                    <small>

                      {uploadedDataset
                        ? `${uploadedDataset.total_tests} uploaded test prompts`
                        : "Upload a dataset from the sidebar"}

                    </small>

                  </div>

                </label>

              </div>

              {!uploadedDataset &&
                !isEvaluating && (

                  <button
                    className="upload-redirect-button"
                    onClick={() =>
                      setActivePage("upload")
                    }
                  >
                    📁 Upload Dataset
                  </button>

                )}

            </div>

            <button
              className="evaluate-button"
              onClick={
                handleStartEvaluation
              }
              disabled={
                isEvaluating ||
                backendStatus !==
                  "connected"
              }
            >

              {isEvaluating
                ? "⏳ Evaluation Running..."
                : "🚀 Start Evaluation"}

            </button>

            {/* LIVE PROGRESS */}

            {isEvaluating && (

              <div className="evaluation-progress-card">

                <div className="evaluation-progress-header">

                  <div>

                    <h3>
                      Evaluation in Progress
                    </h3>

                    <p>
                      Please wait while SentinelLLM
                      evaluates the model.
                    </p>

                  </div>

                  <strong>

                    {progressCurrent} / {progressTotal}

                  </strong>

                </div>

                <div className="evaluation-progress-bar">

                  <div
                    className="evaluation-progress-fill"
                    style={{
                      width:
                        `${progressPercentage}%`,
                    }}
                  >
                  </div>

                </div>

                <div className="evaluation-progress-footer">

                  <span>

                    {progressPercentage}% completed

                  </span>

                  {progressCategory && (

                    <span>

                      Current category:{" "}

                      {
                        formatCategoryName(
                          progressCategory
                        )
                      }

                    </span>

                  )}

                  {progressTest && (

                    <span>

                      Test: {progressTest}

                    </span>

                  )}

                </div>

              </div>

            )}

            {evaluationStatus && (

              <p
                className={
                  evaluationStatus
                    .toLowerCase()
                    .includes("failed") ||
                  evaluationStatus
                    .toLowerCase()
                    .includes("unable") ||
                  evaluationStatus
                    .toLowerCase()
                    .includes("not connected")
                    ? "evaluation-status error"
                    : "evaluation-status"
                }
              >
                {evaluationStatus}
              </p>

            )}

          </section>

        )}

        {/* UPLOAD DATASET PAGE */}

        {activePage === "upload" && (

          <section className="content-card upload-page">

            <h2>
              Upload Your Dataset
            </h2>

            <p>
              Upload a CSV benchmark containing
              prompts, expected answers,
              evaluation criteria, categories,
              and difficulty levels.
            </p>

            <div className="upload-box">

              <input
                type="file"
                accept=".csv"
                disabled={
                  isUploading ||
                  isEvaluating
                }
                onChange={
                  handleDatasetUpload
                }
              />

            </div>

            {isUploading && (

              <div className="loading-message">

                <div className="loading-spinner">
                </div>

                <p>
                  Uploading and validating dataset...
                </p>

              </div>

            )}

            {uploadStatus && (

              <p className="upload-status">
                {uploadStatus}
              </p>

            )}

            {uploadError && (

              <div className="upload-error">

                <p>
                  {uploadError}
                </p>

              </div>

            )}

            {uploadedDataset && (

              <div className="uploaded-info">

                <h3>
                  Dataset Ready
                </h3>

                <p>
                  File:{" "}
                  {
                    uploadedDataset.filename
                  }
                </p>

                <p>
                  Test Cases:{" "}
                  {
                    uploadedDataset.total_tests
                  }
                </p>

                <button
                  className="evaluate-button"
                  onClick={() =>
                    setActivePage(
                      "evaluate"
                    )
                  }
                >
                  🔍 Go to Evaluate Model
                </button>

              </div>

            )}

          </section>

        )}

        {/* HISTORY PAGE */}

        {activePage === "history" && (

          <section
            className="content-card history-page"
          >

            <div
              className="history-heading"
            >

              <div>

                <h2>
                  Evaluation History
                </h2>

                <p>
                  View your previous LLM evaluation
                  runs and safety performance at a glance.
                </p>

              </div>

              <button
                className="refresh-history-button"
                onClick={fetchHistory}
                disabled={historyLoading}
              >
                {historyLoading
                  ? "Loading..."
                  : "↻ Refresh"}
              </button>

            </div>

            {historyLoading ? (

              <div className="history-empty">

                <div className="loading-spinner">
                </div>

                <p>
                  Loading evaluation history...
                </p>

              </div>

            ) : historyError ? (

              <div className="history-empty error-card">

                <h3>
                  Unable to Load History
                </h3>

                <p>
                  {historyError}
                </p>

                <button
                  className="evaluate-button"
                  onClick={fetchHistory}
                >
                  ↻ Retry
                </button>

              </div>

            ) : evaluationHistory.length ===
              0 ? (

              <div className="history-empty">

                <p>
                  No previous evaluations found.
                </p>

              </div>

            ) : (

              <div
                className="history-table-wrapper"
              >

                <table
                  className="history-table"
                >

                  <thead>

                    <tr>

                      <th>
                        Evaluation
                      </th>

                      <th>
                        Model
                      </th>

                      <th>
                        Date & Time
                      </th>

                      <th>
                        Dataset
                      </th>

                      <th>
                        Tests
                      </th>

                      <th>
                        Passed
                      </th>

                      <th>
                        Attention
                      </th>

                      <th>
                        Overall Score
                      </th>

                    </tr>

                  </thead>

                  <tbody>

                    {evaluationHistory.map(
                      (evaluation) => (

                        <tr
                          key={
                            evaluation.evaluation_id
                          }
                        >

                          <td>

                            <span
                              className="evaluation-id"
                            >
                              {
                                evaluation.evaluation_id
                              }
                            </span>

                          </td>

                          <td>

                            <span
                              className="model-name"
                            >
                              {
                                evaluation.model_name
                              }
                            </span>

                          </td>

                          <td>

                            <span
                              className="timestamp"
                            >
                              {
                                evaluation.timestamp
                              }
                            </span>

                          </td>

                          <td>

                            <span
                              className="dataset-badge"
                            >
                              {
                                evaluation.dataset_type
                              }
                            </span>

                          </td>

                          <td>
                            {
                              evaluation.total_tests
                            }
                          </td>

                          <td>

                            <span
                              className="passed-value"
                            >
                              {
                                evaluation.passed_tests
                              }
                            </span>

                          </td>

                          <td>

                            <span
                              className={
                                evaluation.needs_attention > 0
                                  ? "attention-value"
                                  : "attention-value none"
                              }
                            >
                              {
                                evaluation.needs_attention
                              }
                            </span>

                          </td>

                          <td>

                            <span
                              className={
                                evaluation.overall_score >= 80
                                  ? "score-badge good"
                                  : evaluation.overall_score >= 50
                                  ? "score-badge warning"
                                  : "score-badge poor"
                              }
                            >

                              {
                                evaluation.overall_score
                              }

                              /100

                            </span>

                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>

            )}

          </section>

        )}

      </main>

    </div>
  );
}

export default App;