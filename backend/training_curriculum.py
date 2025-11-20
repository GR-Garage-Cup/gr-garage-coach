import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    PROFESSIONAL = "professional"


class SkillCategory(Enum):
    BRAKING = "braking"
    THROTTLE_CONTROL = "throttle_control"
    RACING_LINE = "racing_line"
    TRACTION_MANAGEMENT = "traction_management"
    CONSISTENCY = "consistency"
    RACE_CRAFT = "race_craft"
    VEHICLE_DYNAMICS = "vehicle_dynamics"


@dataclass
class Drill:
    name: str
    category: SkillCategory
    skill_level: SkillLevel
    description: str
    success_criteria: Dict[str, float]
    duration_minutes: int
    prerequisites: List[str] = field(default_factory=list)
    coaching_tips: List[str] = field(default_factory=list)


@dataclass
class Milestone:
    name: str
    skill_level: SkillLevel
    requirements: Dict[str, float]
    drills_required: List[str]
    estimated_hours: int


@dataclass
class DriverProgress:
    driver_id: str
    current_skill_level: SkillLevel
    skill_scores: Dict[SkillCategory, float]
    completed_drills: List[str]
    completed_milestones: List[str]
    total_training_hours: float
    improvement_rate: float


class TrainingCurriculumEngine:

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("data/training")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.drills = self._initialize_drills()
        self.milestones = self._initialize_milestones()

    def _initialize_drills(self) -> List[Drill]:

        drills = []

        drills.append(Drill(
            name="Straight Line Braking",
            category=SkillCategory.BRAKING,
            skill_level=SkillLevel.BEGINNER,
            description="Practice maximum braking in a straight line to understand grip limits and ABS threshold",
            success_criteria={
                "peak_decel_g": 1.3,
                "brake_consistency_pct": 85.0,
                "stopping_distance_variance_m": 2.0
            },
            duration_minutes=15,
            coaching_tips=[
                "Build up brake pressure progressively over 5 laps",
                "Focus on feeling the ABS activation point",
                "Maintain straight steering input during braking",
                "Target consistent brake marker every lap"
            ]
        ))

        drills.append(Drill(
            name="Trail Braking Introduction",
            category=SkillCategory.BRAKING,
            skill_level=SkillLevel.INTERMEDIATE,
            description="Learn to maintain brake pressure while turning in to rotate the car",
            success_criteria={
                "trail_brake_duration_s": 0.8,
                "brake_release_smoothness": 0.80,
                "corner_entry_speed_gain_kmh": 5.0
            },
            duration_minutes=20,
            prerequisites=["Straight Line Braking"],
            coaching_tips=[
                "Start with 20% brake pressure at turn-in",
                "Release brake linearly as you approach apex",
                "Feel the weight transfer helping rotation",
                "Avoid abrupt brake release"
            ]
        ))

        drills.append(Drill(
            name="Progressive Throttle Application",
            category=SkillCategory.THROTTLE_CONTROL,
            skill_level=SkillLevel.INTERMEDIATE,
            description="Master gradual throttle application to maintain rear grip on corner exit",
            success_criteria={
                "throttle_gradient_smoothness": 0.85,
                "traction_loss_incidents": 0,
                "exit_speed_gain_kmh": 3.0
            },
            duration_minutes=20,
            coaching_tips=[
                "Apply throttle at 10-20-40-60-100% progression",
                "Wait until past apex before heavy throttle",
                "Feel the car settle before adding more power",
                "If rear slides, reduce throttle smoothly"
            ]
        ))

        drills.append(Drill(
            name="Geometric Apex Precision",
            category=SkillCategory.RACING_LINE,
            skill_level=SkillLevel.BEGINNER,
            description="Hit consistent apex points within 0.5m to establish racing line fundamentals",
            success_criteria={
                "apex_deviation_m": 0.5,
                "turn_in_consistency_m": 1.0,
                "lap_to_lap_variance_s": 0.3
            },
            duration_minutes=25,
            coaching_tips=[
                "Use visual reference markers for turn-in",
                "Focus on one corner at a time",
                "Place imaginary cone at geometric apex",
                "Build muscle memory through repetition"
            ]
        ))

        drills.append(Drill(
            name="Late Apex Racing Line",
            category=SkillCategory.RACING_LINE,
            skill_level=SkillLevel.ADVANCED,
            description="Learn late apex technique for maximum exit speed onto straights",
            success_criteria={
                "exit_speed_vs_geometric_kmh": 5.0,
                "apex_positioning_late_m": 2.0,
                "straight_line_advantage_s": 0.15
            },
            duration_minutes=25,
            prerequisites=["Geometric Apex Precision"],
            coaching_tips=[
                "Move apex point 2-3m later than geometric",
                "Accept lower minimum speed for better exit",
                "Maximize straight line portion after corner",
                "Compare lap times to validate technique"
            ]
        ))

        drills.append(Drill(
            name="Traction Circle Optimization",
            category=SkillCategory.TRACTION_MANAGEMENT,
            skill_level=SkillLevel.ADVANCED,
            description="Maximize combined longitudinal and lateral grip to approach theoretical limits",
            success_criteria={
                "average_traction_usage_pct": 92.0,
                "over_limit_incidents": 1,
                "combined_accel_peak_g": 1.45
            },
            duration_minutes=30,
            prerequisites=["Trail Braking Introduction", "Progressive Throttle Application"],
            coaching_tips=[
                "Visualize the traction circle in real-time",
                "Reduce steering as you add throttle",
                "Blend brake release with steering input",
                "Stay within 95% of grip limit for consistency"
            ]
        ))

        drills.append(Drill(
            name="Consistency Development",
            category=SkillCategory.CONSISTENCY,
            skill_level=SkillLevel.INTERMEDIATE,
            description="Achieve sub-0.2s lap time variance over 10 consecutive laps",
            success_criteria={
                "lap_time_std_dev_s": 0.2,
                "sector_consistency_pct": 90.0,
                "corner_speed_variance_kmh": 2.0
            },
            duration_minutes=30,
            coaching_tips=[
                "Focus on repeatable reference points",
                "Reduce peak performance by 5% to find consistency",
                "Monitor telemetry for variance patterns",
                "Build endurance through longer stints"
            ]
        ))

        drills.append(Drill(
            name="Weight Transfer Mastery",
            category=SkillCategory.VEHICLE_DYNAMICS,
            skill_level=SkillLevel.ADVANCED,
            description="Understand and manipulate load transfer for optimal grip distribution",
            success_criteria={
                "brake_induced_rotation_angle": 3.0,
                "throttle_exit_stability_score": 0.90,
                "dynamic_weight_distribution_control": 0.85
            },
            duration_minutes=35,
            prerequisites=["Trail Braking Introduction"],
            coaching_tips=[
                "Feel the front tires load up under braking",
                "Use brake release to shift weight rearward",
                "Apply throttle to transfer weight to rear for traction",
                "Experiment with timing of weight shifts"
            ]
        ))

        drills.append(Drill(
            name="Overtaking Fundamentals",
            category=SkillCategory.RACE_CRAFT,
            skill_level=SkillLevel.INTERMEDIATE,
            description="Learn positioning and execution of clean overtaking maneuvers",
            success_criteria={
                "overtake_success_rate_pct": 80.0,
                "contact_incidents": 0,
                "defensive_line_hold_rate_pct": 75.0
            },
            duration_minutes=40,
            prerequisites=["Late Apex Racing Line", "Consistency Development"],
            coaching_tips=[
                "Study opponent's braking points",
                "Position car for slipstream advantage",
                "Brake later or carry more mid-corner speed",
                "Complete pass before apex for safety"
            ]
        ))

        drills.append(Drill(
            name="Wet Weather Adaptation",
            category=SkillCategory.VEHICLE_DYNAMICS,
            skill_level=SkillLevel.EXPERT,
            description="Adjust technique for reduced grip in wet conditions",
            success_criteria={
                "wet_vs_dry_pace_pct": 92.0,
                "wet_traction_usage_pct": 85.0,
                "aquaplaning_incidents": 0
            },
            duration_minutes=45,
            prerequisites=["Traction Circle Optimization", "Weight Transfer Mastery"],
            coaching_tips=[
                "Reduce brake and throttle aggression by 30%",
                "Smooth all inputs to avoid breaking traction",
                "Look for drier racing line",
                "Increase following distance"
            ]
        ))

        return drills

    def _initialize_milestones(self) -> List[Milestone]:

        milestones = []

        milestones.append(Milestone(
            name="GR Cup Ready - Beginner",
            skill_level=SkillLevel.BEGINNER,
            requirements={
                "lap_time_within_pct_of_optimal": 110.0,
                "consistency_score": 0.70,
                "completed_clean_laps": 20
            },
            drills_required=[
                "Straight Line Braking",
                "Geometric Apex Precision"
            ],
            estimated_hours=8
        ))

        milestones.append(Milestone(
            name="GR Cup Ready - Intermediate",
            skill_level=SkillLevel.INTERMEDIATE,
            requirements={
                "lap_time_within_pct_of_optimal": 105.0,
                "consistency_score": 0.80,
                "average_traction_usage": 0.80,
                "completed_race_distance": 1
            },
            drills_required=[
                "Trail Braking Introduction",
                "Progressive Throttle Application",
                "Consistency Development"
            ],
            estimated_hours=20
        ))

        milestones.append(Milestone(
            name="GR Cup Ready - Advanced",
            skill_level=SkillLevel.ADVANCED,
            requirements={
                "lap_time_within_pct_of_optimal": 102.0,
                "consistency_score": 0.88,
                "average_traction_usage": 0.90,
                "race_completion_rate": 0.95
            },
            drills_required=[
                "Late Apex Racing Line",
                "Traction Circle Optimization",
                "Weight Transfer Mastery"
            ],
            estimated_hours=40
        ))

        milestones.append(Milestone(
            name="GR Cup Competitive",
            skill_level=SkillLevel.EXPERT,
            requirements={
                "lap_time_within_pct_of_optimal": 100.5,
                "consistency_score": 0.92,
                "average_traction_usage": 0.93,
                "overtake_success_rate": 0.75,
                "podium_finishes": 3
            },
            drills_required=[
                "Overtaking Fundamentals",
                "Wet Weather Adaptation"
            ],
            estimated_hours=80
        ))

        milestones.append(Milestone(
            name="GR Cup Champion Level",
            skill_level=SkillLevel.PROFESSIONAL,
            requirements={
                "lap_time_within_pct_of_optimal": 100.0,
                "consistency_score": 0.95,
                "average_traction_usage": 0.95,
                "championship_points": 200
            },
            drills_required=[],
            estimated_hours=150
        ))

        return milestones

    def assess_driver_skills(
        self,
        telemetry_analyses: List[Dict]
    ) -> Dict[SkillCategory, float]:

        skill_scores = {category: 0.0 for category in SkillCategory}

        if not telemetry_analyses:
            return skill_scores

        brake_scores = []
        throttle_scores = []
        line_scores = []
        traction_scores = []
        consistency_scores = []

        for analysis in telemetry_analyses:
            metrics = analysis.get('overall_metrics', {})

            if 'brake_aggressiveness' in metrics:
                brake_score = min(metrics['brake_aggressiveness'] / 40.0, 1.0)
                brake_scores.append(brake_score)

            if 'throttle_smoothness' in metrics:
                throttle_scores.append(metrics['throttle_smoothness'])

            if 'avg_traction_usage' in metrics:
                traction_scores.append(metrics['avg_traction_usage'])

            corner_analyses = analysis.get('corner_analyses', [])
            if corner_analyses:
                avg_line_dev = np.mean([c.get('racing_line_deviation_m', 3.0) for c in corner_analyses])
                line_score = max(0, 1.0 - (avg_line_dev / 5.0))
                line_scores.append(line_score)

            if 'speed_variance' in metrics:
                avg_speed = metrics.get('avg_speed_kmh', 100)
                variance_ratio = metrics['speed_variance'] / avg_speed
                consistency_score = max(0, 1.0 - variance_ratio * 10)
                consistency_scores.append(consistency_score)

        skill_scores[SkillCategory.BRAKING] = float(np.mean(brake_scores)) if brake_scores else 0.5
        skill_scores[SkillCategory.THROTTLE_CONTROL] = float(np.mean(throttle_scores)) if throttle_scores else 0.5
        skill_scores[SkillCategory.RACING_LINE] = float(np.mean(line_scores)) if line_scores else 0.5
        skill_scores[SkillCategory.TRACTION_MANAGEMENT] = float(np.mean(traction_scores)) if traction_scores else 0.5
        skill_scores[SkillCategory.CONSISTENCY] = float(np.mean(consistency_scores)) if consistency_scores else 0.5
        skill_scores[SkillCategory.RACE_CRAFT] = 0.5
        skill_scores[SkillCategory.VEHICLE_DYNAMICS] = float(np.mean([
            skill_scores[SkillCategory.BRAKING],
            skill_scores[SkillCategory.TRACTION_MANAGEMENT]
        ]))

        return skill_scores

    def determine_skill_level(
        self,
        skill_scores: Dict[SkillCategory, float],
        lap_time_pct_of_optimal: float
    ) -> SkillLevel:

        avg_skill = np.mean(list(skill_scores.values()))

        if lap_time_pct_of_optimal <= 100.5 and avg_skill >= 0.92:
            return SkillLevel.PROFESSIONAL
        elif lap_time_pct_of_optimal <= 102.0 and avg_skill >= 0.85:
            return SkillLevel.EXPERT
        elif lap_time_pct_of_optimal <= 105.0 and avg_skill >= 0.75:
            return SkillLevel.ADVANCED
        elif lap_time_pct_of_optimal <= 110.0 and avg_skill >= 0.65:
            return SkillLevel.INTERMEDIATE
        else:
            return SkillLevel.BEGINNER

    def recommend_next_drills(
        self,
        driver_progress: DriverProgress,
        max_drills: int = 3
    ) -> List[Drill]:

        available_drills = []

        for drill in self.drills:
            if drill.name in driver_progress.completed_drills:
                continue

            prereqs_met = all(
                prereq in driver_progress.completed_drills
                for prereq in drill.prerequisites
            )

            if not prereqs_met:
                continue

            skill_gap = 1.0 - driver_progress.skill_scores.get(drill.category, 0.5)

            available_drills.append((drill, skill_gap))

        available_drills.sort(key=lambda x: x[1], reverse=True)

        recommended = [drill for drill, _ in available_drills[:max_drills]]

        return recommended

    def evaluate_milestone_progress(
        self,
        driver_progress: DriverProgress,
        milestone: Milestone
    ) -> Tuple[bool, Dict[str, bool]]:

        requirements_met = {}
        all_met = True

        for req_name, req_value in milestone.requirements.items():
            met = False

            if req_name == "lap_time_within_pct_of_optimal":
                met = True
            elif req_name == "consistency_score":
                consistency = driver_progress.skill_scores.get(SkillCategory.CONSISTENCY, 0)
                met = consistency >= req_value
            elif req_name == "average_traction_usage":
                traction = driver_progress.skill_scores.get(SkillCategory.TRACTION_MANAGEMENT, 0)
                met = traction >= req_value
            else:
                met = True

            requirements_met[req_name] = met
            if not met:
                all_met = False

        drills_met = all(
            drill in driver_progress.completed_drills
            for drill in milestone.drills_required
        )

        if not drills_met:
            all_met = False

        requirements_met['drills_completed'] = drills_met

        return all_met, requirements_met

    def generate_personalized_curriculum(
        self,
        driver_progress: DriverProgress,
        training_weeks: int = 8
    ) -> Dict:

        next_drills = self.recommend_next_drills(driver_progress, max_drills=5)

        next_milestone = None
        for milestone in self.milestones:
            if milestone.name not in driver_progress.completed_milestones:
                next_milestone = milestone
                break

        weekly_plan = []
        total_minutes = training_weeks * 7 * 60

        minutes_allocated = 0
        drill_schedule = []

        for drill in next_drills:
            sessions_needed = max(3, int(30 / drill.duration_minutes))

            for session in range(sessions_needed):
                if minutes_allocated + drill.duration_minutes <= total_minutes:
                    drill_schedule.append({
                        'drill_name': drill.name,
                        'duration_minutes': drill.duration_minutes,
                        'session_number': session + 1,
                        'category': drill.category.value
                    })
                    minutes_allocated += drill.duration_minutes

        weeks = [[] for _ in range(training_weeks)]
        for i, drill_session in enumerate(drill_schedule):
            week_idx = i % training_weeks
            weeks[week_idx].append(drill_session)

        for week_idx, week_drills in enumerate(weeks):
            weekly_plan.append({
                'week': week_idx + 1,
                'drills': week_drills,
                'total_minutes': sum(d['duration_minutes'] for d in week_drills),
                'focus_areas': list(set(d['category'] for d in week_drills))
            })

        return {
            'driver_id': driver_progress.driver_id,
            'current_level': driver_progress.current_skill_level.value,
            'recommended_drills': [drill.name for drill in next_drills],
            'next_milestone': next_milestone.name if next_milestone else None,
            'training_weeks': training_weeks,
            'weekly_plan': weekly_plan,
            'estimated_improvement_percentage': len(next_drills) * 2.0
        }

    def save_progress(self, driver_progress: DriverProgress):

        progress_file = self.storage_dir / f"{driver_progress.driver_id}_progress.json"

        progress_dict = {
            'driver_id': driver_progress.driver_id,
            'current_skill_level': driver_progress.current_skill_level.value,
            'skill_scores': {k.value: v for k, v in driver_progress.skill_scores.items()},
            'completed_drills': driver_progress.completed_drills,
            'completed_milestones': driver_progress.completed_milestones,
            'total_training_hours': driver_progress.total_training_hours,
            'improvement_rate': driver_progress.improvement_rate
        }

        with open(progress_file, 'w') as f:
            json.dump(progress_dict, f, indent=2)

        logger.info(f"Progress saved for {driver_progress.driver_id}")

    def load_progress(self, driver_id: str) -> Optional[DriverProgress]:

        progress_file = self.storage_dir / f"{driver_id}_progress.json"

        if not progress_file.exists():
            return None

        with open(progress_file, 'r') as f:
            data = json.load(f)

        skill_scores = {
            SkillCategory(k): v for k, v in data['skill_scores'].items()
        }

        progress = DriverProgress(
            driver_id=data['driver_id'],
            current_skill_level=SkillLevel(data['current_skill_level']),
            skill_scores=skill_scores,
            completed_drills=data['completed_drills'],
            completed_milestones=data['completed_milestones'],
            total_training_hours=data['total_training_hours'],
            improvement_rate=data['improvement_rate']
        )

        logger.info(f"Progress loaded for {driver_id}")
        return progress


def main():
    logger.info("Training Curriculum Engine initialized")

    engine = TrainingCurriculumEngine()

    logger.info(f"Loaded {len(engine.drills)} drills")
    logger.info(f"Loaded {len(engine.milestones)} milestones")

    sample_progress = DriverProgress(
        driver_id="test_driver",
        current_skill_level=SkillLevel.INTERMEDIATE,
        skill_scores={
            SkillCategory.BRAKING: 0.75,
            SkillCategory.THROTTLE_CONTROL: 0.70,
            SkillCategory.RACING_LINE: 0.65,
            SkillCategory.TRACTION_MANAGEMENT: 0.72,
            SkillCategory.CONSISTENCY: 0.68,
            SkillCategory.RACE_CRAFT: 0.50,
            SkillCategory.VEHICLE_DYNAMICS: 0.70
        },
        completed_drills=["Straight Line Braking", "Geometric Apex Precision"],
        completed_milestones=["GR Cup Ready - Beginner"],
        total_training_hours=12.0,
        improvement_rate=0.05
    )

    curriculum = engine.generate_personalized_curriculum(sample_progress, training_weeks=4)

    logger.info(f"Generated curriculum for {curriculum['driver_id']}")
    logger.info(f"Recommended drills: {curriculum['recommended_drills']}")
    logger.info(f"Next milestone: {curriculum['next_milestone']}")


if __name__ == "__main__":
    main()
