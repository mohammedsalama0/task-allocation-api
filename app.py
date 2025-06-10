import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings
import os

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Task Allocation API", description="API for assigning tasks to employees using ML")

# Define input model for the API endpoint
class TaskInput(BaseModel):
    complexity: str  # Task complexity: 'High', 'Low', 'Medium'
    skill_indices: list[int] | None  # List of skill indices or None for no skills

# Global variables to store model, scaler, and related data
model = None
scaler = None
feature_columns = None
employee_ids = None
numerical_cols = None
all_skills = None
assigned_employees = {}  # Track assigned employees per task

# Preprocess data (unchanged from original)
def preprocess_data(df):
    all_skills = set()
    df['employee_skills'] = df['employee_skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    for skills in df['employee_skills']:
        all_skills.update(skill.strip() for skill in skills)
    for skill in all_skills:
        df[f'has_{skill}'] = df['employee_skills'].apply(lambda x: 1 if skill in x else 0)
    df = df.drop('employee_skills', axis=1)

    categorical_cols = [col for col in df.columns if col.startswith('department_') or col.startswith('position_') or col.startswith('experience_level_') or col.startswith('dept_position_') or col.startswith('task_complexity_')]
    for col in categorical_cols:
        df[col] = df[col].astype(int)

    numerical_cols = [
        'years_experience', 'current_tasks', 'max_capacity', 'urgent_tasks', 'completed_tasks',
        'total_tasks', 'task_completion_rate', 'accuracy_score', 'feedback_score',
        'available_hours', 'total_hours', 'base_salary', 'performance_bonus',
        'total_compensation', 'skill_match_score', 'availability_score', 'performance_score',
        'workload_balance', 'deadline_pressure', 'market_value_score', 'urgent_task_ratio',
        'workload_ratio', 'experience_salary_ratio', 'composite_performance', 'task_success'
    ]
    scaler = StandardScaler()
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler, numerical_cols, all_skills

# Train model (unchanged from original)
def train_model(df, scaler, numerical_cols):
    employee_ids = df['employee_id']
    df = df.drop('employee_id', axis=1)
    X = df.drop('task_assignment', axis=1)
    y = df['task_assignment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf, X.columns, employee_ids, scaler, numerical_cols

# Prepare task features (unchanged from original)
def prepare_task_features(df, task_complexity, selected_skills, feature_columns, scaler, numerical_cols):
    task_df = df.copy()
    task_df['task_complexity_High'] = 1 if task_complexity == 'High' else 0
    task_df['task_complexity_Low'] = 1 if task_complexity == 'Low' else 0
    task_df['task_complexity_Medium'] = 1 if task_complexity == 'Medium' else 0
    for col in task_df.columns:
        if col.startswith('has_'):
            skill_name = col.replace('has_', '')
            task_df[col] = 1 if skill_name in selected_skills else 0
    for col in feature_columns:
        if col not in task_df.columns:
            task_df[col] = 0
    task_X = task_df[feature_columns]
    task_X[numerical_cols] = scaler.transform(task_X[numerical_cols])
    return task_X

# Load and train model at startup
@app.on_event("startup")
async def startup_event():
    global model, scaler, feature_columns, employee_ids, numerical_cols, all_skills
    # Load dataset from /data/ folder
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'balanced_simple_hr_ai_dataset.csv')
    if not os.path.exists(dataset_path):
        raise RuntimeError("Dataset file not found")
    df = pd.read_csv(dataset_path)
    df, scaler, numerical_cols, all_skills = preprocess_data(df)
    model, feature_columns, employee_ids, scaler, numerical_cols = train_model(df, scaler, numerical_cols)

# Endpoint to get available skills
@app.get("/skills")
async def get_skills():
    return {"skills": sorted(list(all_skills))}

# Endpoint to assign employees to a task
@app.post("/assign-task")
async def assign_task(task: TaskInput):
    global model, scaler, feature_columns, employee_ids, numerical_cols, all_skills, assigned_employees
    # Validate task complexity
    valid_complexities = ['High', 'Low', 'Medium']
    if task.complexity not in valid_complexities:
        raise HTTPException(status_code=400, detail=f"Invalid complexity. Must be one of {valid_complexities}")

    # Validate skill indices
    sorted_skills = sorted(all_skills)
    selected_skills = []
    if task.skill_indices:
        try:
            for idx in task.skill_indices:
                if idx < 1 or idx > len(sorted_skills):
                    raise HTTPException(status_code=400, detail=f"Invalid skill index: {idx}. Must be between 1 and {len(sorted_skills)}")
                selected_skills.append(sorted_skills[idx - 1])
        except ValueError:
            raise HTTPException(status_code=400, detail="Skill indices must be integers")

    # Create task key for tracking assignments
    task_key = (task.complexity, tuple(sorted(selected_skills)))
    if task_key not in assigned_employees:
        assigned_employees[task_key] = set()

    # Load dataset again for task features
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'balanced_simple_hr_ai_dataset.csv')
    df = pd.read_csv(dataset_path)
    df, _, _, _ = preprocess_data(df)  # Preprocess without saving new scaler/all_skills

    # Prepare task features
    task_X = prepare_task_features(df, task.complexity, selected_skills, feature_columns, scaler, numerical_cols)
    probas = model.predict_proba(task_X)[:, 1]
    adjusted_probas = np.clip(probas * 1.20, 0, 1)  # Adjust confidence by 20%
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'employee_id': employee_ids,
        'confidence': adjusted_probas
    })
    predictions['confidence'] = predictions['confidence'].round(4)
    predictions = predictions[~predictions['employee_id'].isin(assigned_employees[task_key])]
    
    # Get top 10 employees
    top_10 = predictions.nlargest(10, 'confidence')
    if top_10.empty:
        raise HTTPException(status_code=404, detail="No more employees available for this task")

    # Update assigned employees
    assigned_employees[task_key].update(top_10['employee_id'])

    # Format response
    result = {
        "task_complexity": task.complexity,
        "skills": selected_skills or [],
        "employees": top_10[['employee_id', 'confidence']].to_dict(orient='records')
    }
    return result