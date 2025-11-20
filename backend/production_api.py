from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import logging
from dataclasses import asdict
from datetime import datetime

from config import PROCESSED_DATA_DIR, MODELS_DIR, TRD_TRACKS
from trd_data_ingestion import TRDDataParser, TRACK_DATABASE
from vehicle_dynamics import VehicleDynamicsAnalyzer, VehicleParameters
from optimal_racing_line import OptimalRacingLineSolver, RacingLineDatabase
from driver_behavior_ml import AdvancedDriverProfiler
from advanced_coaching_engine import ProfessionalCoachingEngine
from format_converter import TelemetryFormatConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GR Garage Coach API",
    description="Professional racing telemetry analysis and coaching system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    track_name: str = Field(..., description="Track name (e.g., road_america)")
    driver_id: Optional[str] = Field("driver", description="Driver identifier")


class CornerRecommendation(BaseModel):
    corner_id: int
    corner_name: str
    time_loss_seconds: float
    primary_issue: str
    recommendations: List[str]


class LapAnalysisResponse(BaseModel):
    track_name: str
    driver_id: str
    lap_time: float
    optimal_lap_time: float
    time_delta: float

    driver_archetype: str
    driver_confidence: float
    driver_strengths: List[str]
    driver_weaknesses: List[str]

    overall_metrics: Dict[str, float]

    top_improvements: List[CornerRecommendation]
    coaching_priorities: List[str]
    skill_gaps: List[str]
    training_curriculum: List[str]

    visualization_data: Optional[Dict] = None


class TrackListResponse(BaseModel):
    tracks: List[str]


@app.get("/")
async def root():
    return {
        "service": "GR Garage Coach API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Vehicle dynamics analysis (tire slip, load transfer, aero)",
            "Optimal racing line computation",
            "Deep learning driver behavior classification",
            "Physics-based coaching recommendations",
            "Progressive training curriculum"
        ]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/v1/tracks", response_model=TrackListResponse)
async def list_tracks():
    return TrackListResponse(tracks=TRD_TRACKS)


# Global storage for last analysis (for visualization endpoints)
_last_analysis_cache = {}

@app.post("/api/v1/analyze", response_model=LapAnalysisResponse)
async def analyze_telemetry(
    file: UploadFile = File(...),
    track_name: str = "road_america",
    driver_id: str = "driver"
):

    logger.info(f"Received analysis request: track={track_name}, driver={driver_id}")

    if track_name not in TRD_TRACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown track: {track_name}. Available: {TRD_TRACKS}"
        )

    try:
        content = await file.read()

        logger.info(f"Processing uploaded file: {file.filename}")

        converter = TelemetryFormatConverter()
        telemetry_df = converter.convert_to_wide_format(content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            telemetry_df.to_csv(tmp_file.name, index=False)
            tmp_path = Path(tmp_file.name)

        trd_parser = TRDDataParser(track_name=track_name)
        telemetry_df = trd_parser.parse_csv(tmp_path)

        # Extract fastest lap
        lap_number, telemetry_df = trd_parser.get_fastest_lap(telemetry_df)

        logger.info(f"Loaded {len(telemetry_df)} telemetry points (lap {lap_number})")

        coaching_engine = ProfessionalCoachingEngine(
            track_name=track_name,
            models_dir=MODELS_DIR
        )

        report = coaching_engine.analyze_full_lap(
            driver_telemetry=telemetry_df,
            driver_id=driver_id
        )

        # Cache telemetry for visualization endpoints
        _last_analysis_cache[driver_id] = {
            'track_name': track_name,
            'telemetry': telemetry_df,
            'report': report
        }

        top_improvements = [
            CornerRecommendation(
                corner_id=imp['corner_id'],
                corner_name=imp['corner_name'],
                time_loss_seconds=imp['time_loss'],
                primary_issue=imp['primary_issue'],
                recommendations=imp['recommendations']
            )
            for imp in report.key_improvements
        ]

        viz_data = {
            'track_map': {
                'x': telemetry_df['x'].tolist() if 'x' in telemetry_df.columns else [],
                'y': telemetry_df['y'].tolist() if 'y' in telemetry_df.columns else [],
                'arc_length': telemetry_df['arc_length'].tolist() if 'arc_length' in telemetry_df.columns else [],
            },
            'speed_trace': {
                'arc_length': telemetry_df['arc_length'].tolist() if 'arc_length' in telemetry_df.columns else [],
                'speed': telemetry_df['speed'].tolist() if 'speed' in telemetry_df.columns else [],
            },
            'input_traces': {
                'arc_length': telemetry_df['arc_length'].tolist() if 'arc_length' in telemetry_df.columns else [],
                'throttle': telemetry_df['throttle'].tolist() if 'throttle' in telemetry_df.columns else (telemetry_df['ath'].tolist() if 'ath' in telemetry_df.columns else []),
                'brake_front': telemetry_df['brake_front'].tolist() if 'brake_front' in telemetry_df.columns else [],
                'brake_rear': telemetry_df['brake_rear'].tolist() if 'brake_rear' in telemetry_df.columns else [],
            },
            'traction_circle': {
                'longitudinal_g': telemetry_df['accel_x'].tolist() if 'accel_x' in telemetry_df.columns else (telemetry_df['accx_can'].tolist() if 'accx_can' in telemetry_df.columns else []),
                'lateral_g': telemetry_df['accel_y'].tolist() if 'accel_y' in telemetry_df.columns else (telemetry_df['accy_can'].tolist() if 'accy_can' in telemetry_df.columns else []),
            },
            'data_points': len(telemetry_df)
        }

        response = LapAnalysisResponse(
            track_name=report.track_name,
            driver_id=report.driver_id,
            lap_time=report.lap_time,
            optimal_lap_time=report.optimal_lap_time,
            time_delta=report.time_delta,
            driver_archetype=report.driver_style.archetype,
            driver_confidence=report.driver_style.confidence,
            driver_strengths=report.driver_style.strengths,
            driver_weaknesses=report.driver_style.weaknesses,
            overall_metrics=report.overall_metrics,
            top_improvements=top_improvements,
            coaching_priorities=report.coaching_priority_order,
            skill_gaps=report.skill_gaps,
            training_curriculum=report.training_recommendations,
            visualization_data=viz_data
        )

        tmp_path.unlink()

        logger.info("Analysis complete")

        return response

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/batch-process")
async def batch_process_track(
    track_name: str,
    background_tasks: BackgroundTasks
):

    if track_name not in TRD_TRACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown track: {track_name}"
        )

    def process_track_data():
        logger.info(f"Starting batch processing for {track_name}")

        ingestion = TelemetryIngestion()
        ingestion.process_track_directory(track_name)

        logger.info(f"Batch processing complete for {track_name}")

    background_tasks.add_task(process_track_data)

    return {
        "status": "processing",
        "track": track_name,
        "message": "Batch processing started in background"
    }


@app.get("/api/v1/driver-profile/{driver_id}")
async def get_driver_profile(driver_id: str, track_name: Optional[str] = None):

    if track_name and track_name not in TRD_TRACKS:
        raise HTTPException(status_code=400, detail=f"Unknown track: {track_name}")

    return {
        "driver_id": driver_id,
        "track": track_name,
        "message": "Driver profile retrieval not yet implemented - coming soon"
    }


@app.post("/api/v1/compare")
async def compare_laps(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    track_name: str = "road_america"
):

    return {
        "message": "Lap comparison feature coming soon",
        "track": track_name
    }


@app.get("/api/v1/optimal-line/{track_name}")
async def get_optimal_line(track_name: str):

    if track_name not in TRD_TRACKS:
        raise HTTPException(status_code=400, detail=f"Unknown track: {track_name}")

    line_db = RacingLineDatabase(MODELS_DIR)
    optimal_line = line_db.load_optimal_line(track_name)

    if optimal_line is None:
        raise HTTPException(
            status_code=404,
            detail=f"No optimal line computed for {track_name}"
        )

    return {
        "track_name": track_name,
        "data_points": len(optimal_line),
        "data": optimal_line.to_dict(orient='records')
    }


@app.get("/api/v1/metrics/summary")
async def get_platform_metrics():

    return {
        "total_analyses": 0,
        "active_drivers": 0,
        "tracks_available": len(TRD_TRACKS),
        "average_improvement_seconds": 0.0,
        "message": "Metrics tracking coming soon"
    }


@app.get("/api/v1/visualization/{driver_id}")
async def get_visualization_data(driver_id: str):
    """
    Get visualization data for charts/maps after analysis.

    Returns REAL telemetry data including:
    - GPS coordinates (x, y) for track map
    - Speed, brake, throttle traces by arc_length
    - G-force data for traction circle
    """
    if driver_id not in _last_analysis_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for driver {driver_id}. Run /analyze first."
        )

    cache = _last_analysis_cache[driver_id]
    telemetry = cache['telemetry']

    # Prepare visualization data
    viz_data = {
        'track_name': cache['track_name'],
        'driver_id': driver_id,

        # Track map data (GPS coordinates)
        'track_map': {
            'x': telemetry['x'].tolist() if 'x' in telemetry.columns else [],
            'y': telemetry['y'].tolist() if 'y' in telemetry.columns else [],
            'arc_length': telemetry['arc_length'].tolist() if 'arc_length' in telemetry.columns else [],
        },

        # Speed trace (by distance)
        'speed_trace': {
            'arc_length': telemetry['arc_length'].tolist() if 'arc_length' in telemetry.columns else [],
            'speed': telemetry['speed'].tolist() if 'speed' in telemetry.columns else [],
        },

        # Brake/throttle traces
        'input_traces': {
            'arc_length': telemetry['arc_length'].tolist() if 'arc_length' in telemetry.columns else [],
            'throttle': telemetry['throttle'].tolist() if 'throttle' in telemetry.columns else (telemetry['ath'].tolist() if 'ath' in telemetry.columns else []),
            'brake_front': telemetry['brake_front'].tolist() if 'brake_front' in telemetry.columns else [],
            'brake_rear': telemetry['brake_rear'].tolist() if 'brake_rear' in telemetry.columns else [],
        },

        # Traction circle (G-G diagram)
        'traction_circle': {
            'longitudinal_g': telemetry['accel_x'].tolist() if 'accel_x' in telemetry.columns else (telemetry['accx_can'].tolist() if 'accx_can' in telemetry.columns else []),
            'lateral_g': telemetry['accel_y'].tolist() if 'accel_y' in telemetry.columns else (telemetry['accy_can'].tolist() if 'accy_can' in telemetry.columns else []),
        },

        # Data points count
        'data_points': len(telemetry)
    }

    return viz_data


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting GR Garage Coach API Server")

    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
