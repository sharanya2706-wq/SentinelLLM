import shutil
import uuid
import threading
import logging
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, SecretStr

from evaluation.runner import run_evaluation

from evaluation.history_storage import (
    load_history,
    save_evaluation_history,
)

from utils.dataset_loader import load_benchmark


# ========================================
# LOGGING
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("sentinellm")


# ========================================
# CREATE FASTAPI APPLICATION
# ========================================

app = FastAPI(
    title="SentinelLLM API",
    description=(
        "LLM Safety and Reliability "
        "Evaluation Framework"
    ),
    version="1.0.0",
)


# ========================================
# CORS
# ========================================

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ========================================
# PROJECT PATHS
# ========================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

UPLOADS_DIR = BASE_DIR / "uploads"

RESULTS_DIR = BASE_DIR / "results"

UPLOADS_DIR.mkdir(
    exist_ok=True
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ========================================
# STORE ALL EVALUATIONS
# ========================================

evaluations = {}

evaluations_lock = threading.Lock()


# ========================================
# STORE STOP EVENTS
# ========================================

"""
Each evaluation receives its own threading.Event.

When the frontend presses "Stop Evaluation",
the corresponding event is set.

The runner checks this event between test cases
and while waiting between requests.
"""

evaluation_stop_events = {}


# ========================================
# REQUEST MODEL
# ========================================

class EvaluationRequest(BaseModel):

    endpoint: str

    # SecretStr prevents the API key from
    # accidentally appearing in normal logs
    # or representations of the request object.
    api_key: SecretStr

    model_name: str

    dataset_type: str = "default"

    filename: str | None = None

    test_mode: bool = False


# ========================================
# SAFE ERROR MESSAGE
# ========================================

def get_safe_error_message(error):

    """
    Prevent sensitive information such as API
    keys from accidentally being returned to
    the frontend.
    """

    message = str(error)

    return message


# ========================================
# HOME
# ========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to SentinelLLM API",
        "status": "running",
    }


# ========================================
# HEALTH
# ========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "SentinelLLM Backend",
    }


# ========================================
# DATASET SUMMARY
# ========================================

@app.get("/dataset/summary")
def dataset_summary():

    dataset_path = (
        BASE_DIR
        / "data"
        / "benchmark.csv"
    )

    try:

        dataset = load_benchmark(
            dataset_path
        )

        category_counts = (
            dataset["category"]
            .value_counts()
            .to_dict()
        )

        difficulty_counts = (
            dataset["difficulty"]
            .value_counts()
            .to_dict()
        )

        return {
            "total_tests": len(dataset),
            "categories": category_counts,
            "difficulty": difficulty_counts,
        }

    except Exception:

        logger.exception(
            "Dataset summary failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load dataset summary."
            ),
        )


# ========================================
# UPLOAD CUSTOM DATASET
# ========================================

@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "A valid file is required."
            ),
        )

    # Remove any directory information from
    # the uploaded filename.
    safe_filename = Path(
        file.filename
    ).name

    if not safe_filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV files are supported."
            ),
        )

    unique_filename = (
        f"{uuid.uuid4()}_{safe_filename}"
    )

    file_path = (
        UPLOADS_DIR
        / unique_filename
    )

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        dataset = load_benchmark(
            file_path
        )

        return {
            "message":
                "Dataset uploaded successfully.",

            "filename":
                unique_filename,

            "total_tests":
                len(dataset),

            "categories":
                dataset["category"]
                .value_counts()
                .to_dict(),
        }

    except Exception:

        logger.exception(
            "Dataset upload failed"
        )

        if file_path.exists():

            file_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid dataset format."
            ),
        )

    finally:

        await file.close()


# ========================================
# EVALUATION HISTORY
# ========================================

@app.get("/history")
def get_evaluation_history():

    try:

        history = load_history()

        return {
            "total_evaluations":
                len(history),

            "history":
                history,
        }

    except Exception:

        logger.exception(
            "Unable to load evaluation history"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load evaluation history."
            ),
        )


# ========================================
# LATEST EVALUATION
# ========================================

@app.get("/history/latest")
def get_latest_evaluation():

    try:

        history = load_history()

        if not history:

            return {
                "message":
                    "No evaluation history found.",

                "evaluation":
                    None,
            }

        return {
            "evaluation":
                history[0],
        }

    except Exception:

        logger.exception(
            "Unable to load latest evaluation"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load latest evaluation."
            ),
        )


# ========================================
# BACKGROUND EVALUATION WORKER
# ========================================

def run_evaluation_background(
    evaluation_id,
    dataset_path,
    endpoint,
    api_key,
    model_name,
    dataset_type,
    test_mode,
    stop_event,
):

    try:

        # ------------------------------------
        # UPDATE STATUS
        # ------------------------------------

        with evaluations_lock:

            if (
                evaluation_id
                not in evaluations
            ):
                return

            evaluations[
                evaluation_id
            ].update({
                "status": "running",
            })

        # ------------------------------------
        # PROGRESS CALLBACK
        # ------------------------------------

        def update_progress(
            progress_data
        ):

            with evaluations_lock:

                if (
                    evaluation_id
                    in evaluations
                ):

                    # Do not overwrite a stopped
                    # state with a late running update.
                    current_status = (
                        evaluations[
                            evaluation_id
                        ].get("status")
                    )

                    if current_status == "stopped":
                        return

                    evaluations[
                        evaluation_id
                    ].update(
                        progress_data
                    )

        # ------------------------------------
        # RUN EVALUATION
        # ------------------------------------

        evaluation_results = run_evaluation(
            dataset_path=dataset_path,
            endpoint=endpoint,
            api_key=api_key,
            model_name=model_name,
            test_mode=test_mode,
            progress_callback=update_progress,
            stop_event=stop_event,
        )

        # ------------------------------------
        # CHECK WHETHER USER STOPPED IT
        # ------------------------------------

        if stop_event.is_set():

            with evaluations_lock:

                if (
                    evaluation_id
                    in evaluations
                ):

                    evaluations[
                        evaluation_id
                    ].update({
                        "status": "stopped",

                        "completed":
                            len(evaluation_results),

                        "total":
                            (
                                evaluations[
                                    evaluation_id
                                ].get(
                                    "total",
                                    len(evaluation_results)
                                )
                            ),

                        "percentage":
                            (
                                round(
                                    (
                                        len(evaluation_results)
                                        /
                                        max(
                                            1,
                                            evaluations[
                                                evaluation_id
                                            ].get(
                                                "total",
                                                len(evaluation_results)
                                            )
                                        )
                                    ) * 100,
                                    2
                                )
                            ),

                        "current_test":
                            None,

                        "current_category":
                            None,

                        "history_entry":
                            None,

                        "results":
                            evaluation_results,
                    })

            logger.info(
                "Evaluation %s stopped by user after %s completed test cases.",
                evaluation_id,
                len(evaluation_results),
            )

            return

        # ------------------------------------
        # SAVE HISTORY
        # ------------------------------------

        # Only completed evaluations are saved
        # to evaluation history.
        history_entry = save_evaluation_history(
            model_name=model_name,

            dataset_type=(
                "test"
                if test_mode
                else dataset_type
            ),

            results=evaluation_results,
        )

        # ------------------------------------
        # MARK COMPLETED
        # ------------------------------------

        with evaluations_lock:

            if (
                evaluation_id
                in evaluations
            ):

                evaluations[
                    evaluation_id
                ].update({
                    "status": "completed",

                    "completed":
                        len(evaluation_results),

                    "total":
                        len(evaluation_results),

                    "percentage":
                        100,

                    "current_test":
                        None,

                    "current_category":
                        None,

                    "history_entry":
                        history_entry,

                    "results":
                        evaluation_results,
                })

    except Exception as e:

        logger.exception(
            "Evaluation %s failed",
            evaluation_id,
        )

        with evaluations_lock:

            if evaluation_id in evaluations:

                evaluations[
                    evaluation_id
                ].update({
                    "status": "failed",

                    "error":
                        get_safe_error_message(
                            e
                        ),
                })


# ========================================
# START EVALUATION
# ========================================

@app.post("/evaluate")
def evaluate_model(
    request: EvaluationRequest
):

    try:

        # ------------------------------------
        # VALIDATE MODEL CONFIGURATION
        # ------------------------------------

        endpoint = request.endpoint.strip()

        # Extract the secret value only when
        # we actually need to call the model.
        #
        # Never log this variable.
        api_key = (
            request.api_key
            .get_secret_value()
            .strip()
        )

        model_name = (
            request.model_name
            .strip()
        )

        if not endpoint:

            raise HTTPException(
                status_code=400,
                detail=(
                    "API endpoint is required."
                ),
            )

        if not api_key:

            raise HTTPException(
                status_code=400,
                detail=(
                    "API key is required."
                ),
            )

        if not model_name:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Model name is required."
                ),
            )

        # ------------------------------------
        # SELECT DATASET
        # ------------------------------------

        if request.dataset_type == "default":

            dataset_path = (
                BASE_DIR
                / "data"
                / "benchmark.csv"
            )

            if not dataset_path.exists():

                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Default benchmark "
                        "dataset not found."
                    ),
                )

        elif request.dataset_type == "custom":

            if not request.filename:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Custom dataset filename "
                        "is required."
                    ),
                )

            # Prevent path traversal.
            safe_filename = Path(
                request.filename
            ).name

            dataset_path = (
                UPLOADS_DIR
                / safe_filename
            )

            if not dataset_path.exists():

                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Uploaded dataset "
                        "not found."
                    ),
                )

        else:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid dataset type. "
                    "Use 'default' or 'custom'."
                ),
            )

        # ------------------------------------
        # CREATE EVALUATION ID
        # ------------------------------------

        evaluation_id = str(
            uuid.uuid4()
        )

        # ------------------------------------
        # CREATE STOP EVENT
        # ------------------------------------

        stop_event = threading.Event()

        # ------------------------------------
        # CREATE EVALUATION RECORD
        # ------------------------------------

        # IMPORTANT:
        # Never store the API key inside the
        # evaluations dictionary.
        with evaluations_lock:

            evaluations[
                evaluation_id
            ] = {
                "evaluation_id":
                    evaluation_id,

                "status":
                    "starting",

                "completed":
                    0,

                "total":
                    0,

                "percentage":
                    0,

                "current_test":
                    None,

                "current_category":
                    None,

                "history_entry":
                    None,

                "error":
                    None,
            }

            evaluation_stop_events[
                evaluation_id
            ] = stop_event

        # ------------------------------------
        # START BACKGROUND THREAD
        # ------------------------------------

        evaluation_thread = threading.Thread(
            target=run_evaluation_background,

            args=(
                evaluation_id,
                dataset_path,
                endpoint,
                api_key,
                model_name,
                request.dataset_type,
                request.test_mode,
                stop_event,
            ),

            daemon=True,
        )

        evaluation_thread.start()

        # ------------------------------------
        # RETURN SAFE RESPONSE
        # ------------------------------------

        return {
            "message":
                "Evaluation started successfully.",

            "evaluation_id":
                evaluation_id,
        }

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "Unable to start evaluation"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to start evaluation."
            ),
        )


# ========================================
# STOP EVALUATION
# ========================================

@app.post(
    "/stop-evaluation/{evaluation_id}"
)
def stop_evaluation(
    evaluation_id: str
):

    with evaluations_lock:

        evaluation = evaluations.get(
            evaluation_id
        )

        if not evaluation:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Evaluation not found."
                ),
            )

        status = evaluation.get(
            "status"
        )

        # ------------------------------------
        # ALREADY STOPPED
        # ------------------------------------

        if status == "stopped":

            return {
                "message":
                    "Evaluation has already been stopped.",

                "evaluation_id":
                    evaluation_id,

                "status":
                    "stopped",
            }

        # ------------------------------------
        # ALREADY COMPLETED
        # ------------------------------------

        if status == "completed":

            return {
                "message":
                    "Evaluation has already completed.",

                "evaluation_id":
                    evaluation_id,

                "status":
                    "completed",
            }

        # ------------------------------------
        # ALREADY FAILED
        # ------------------------------------

        if status == "failed":

            return {
                "message":
                    "Evaluation has already failed.",

                "evaluation_id":
                    evaluation_id,

                "status":
                    "failed",
            }

        # ------------------------------------
        # GET STOP EVENT
        # ------------------------------------

        stop_event = (
            evaluation_stop_events.get(
                evaluation_id
            )
        )

        if not stop_event:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Unable to stop this evaluation."
                ),
            )

        # ------------------------------------
        # REQUEST STOP
        # ------------------------------------

        stop_event.set()

        evaluation.update({
            "status":
                "stopping",
        })

        logger.info(
            "Stop requested for evaluation %s",
            evaluation_id,
        )

        return {
            "message":
                "Stop request received. The evaluation will stop after the current test finishes.",

            "evaluation_id":
                evaluation_id,

            "status":
                "stopping",
        }


# ========================================
# GET EVALUATION PROGRESS
# ========================================

@app.get(
    "/evaluation-progress/{evaluation_id}"
)
def get_evaluation_progress(
    evaluation_id: str
):

    with evaluations_lock:

        evaluation = evaluations.get(
            evaluation_id
        )

        if not evaluation:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Evaluation not found."
                ),
            )

        # The evaluation object does not contain
        # the API key, so it is safe to return.
        return evaluation