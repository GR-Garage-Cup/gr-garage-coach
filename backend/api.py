from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io
import logging

from data_ingestion import TelemetryIngestion
from coaching_engine import CoachingEngine
from driver_dna import DriverDNA
from config import TRD_TRACKS
from format_converter import TelemetryFormatConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GR Garage Coach API",
    description="Championship coaching from TRD telemetry data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    track: str


class AnalysisResponse(BaseModel):
    track: str
    average_speed_delta: float
    corner_issues: List[Dict]
    recommendations: List[str]
    driver_dna: Optional[Dict] = None


class TracksResponse(BaseModel):
    tracks: List[str]


@app.get("/")
def read_root():
    return {
        "service": "GR Garage Coach API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/tracks", response_model=TracksResponse)
def get_available_tracks():
    return TracksResponse(tracks=TRD_TRACKS)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_lap(
    file: UploadFile = File(...),
    track: str = "road_america"
):
    logger.info(f"Received analysis request for track: {track}")

    if track not in TRD_TRACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid track. Must be one of: {', '.join(TRD_TRACKS)}"
        )

    try:
        contents = await file.read()

        converter = TelemetryFormatConverter()
        df = converter.convert_to_wide_format(contents)

        logger.info(f"Loaded telemetry with {len(df)} rows")

    except ValueError as e:
        logger.error(f"Error converting file format: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error converting file format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error reading CSV file: {str(e)}"
        )

    try:
        ingestion = TelemetryIngestion()
        df = ingestion.project_gps_to_xy(df)
        df = ingestion.normalize_telemetry(df)
        df = ingestion.compute_arc_length(df)

        logger.info("Telemetry preprocessing complete")

    except Exception as e:
        logger.error(f"Error preprocessing telemetry: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error preprocessing telemetry: {str(e)}"
        )

    try:
        coach = CoachingEngine(track)
        analysis = coach.analyze_lap(df)

        recommendations = coach.generate_coaching_recommendations(analysis)

        logger.info("Lap analysis complete")

    except Exception as e:
        logger.error(f"Error analyzing lap: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing lap: {str(e)}"
        )

    driver_dna_result = None
    try:
        dna_engine = DriverDNA()
        driver_dna_result = dna_engine.classify_driver(df, track)
        logger.info("Driver DNA classification complete")
    except Exception as e:
        logger.warning(f"Driver DNA classification failed: {e}")

    return AnalysisResponse(
        track=track,
        average_speed_delta=float(analysis.get('average_speed_delta', 0.0)),
        corner_issues=[
            {
                'corner_id': int(issue['corner_id']),
                'brake_analysis': issue.get('brake_analysis', {}),
                'throttle_analysis': issue.get('throttle_analysis', {})
            }
            for issue in analysis.get('corner_issues', [])
        ],
        recommendations=recommendations,
        driver_dna=driver_dna_result
    )


@app.get("/baseline/{track}")
def get_baseline(track: str):
    logger.info(f"Fetching baseline for track: {track}")

    if track not in TRD_TRACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid track. Must be one of: {', '.join(TRD_TRACKS)}"
        )

    try:
        from pathlib import Path
        baseline_path = Path(f"../data/models/{track}/champion_baseline.parquet")

        if not baseline_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Baseline not found for track: {track}"
            )

        baseline_df = pd.read_parquet(baseline_path)

        return {
            "track": track,
            "num_points": len(baseline_df),
            "data": baseline_df.to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"Error fetching baseline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching baseline: {str(e)}"
        )


@app.get("/corners/{track}")
def get_corners(track: str):
    logger.info(f"Fetching corners for track: {track}")

    if track not in TRD_TRACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid track. Must be one of: {', '.join(TRD_TRACKS)}"
        )

    try:
        from pathlib import Path
        corners_path = Path(f"../data/models/{track}/corners_metadata.parquet")

        if not corners_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Corners metadata not found for track: {track}"
            )

        corners_df = pd.read_parquet(corners_path)

        return {
            "track": track,
            "num_corners": len(corners_df),
            "corners": corners_df.to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"Error fetching corners: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching corners: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
