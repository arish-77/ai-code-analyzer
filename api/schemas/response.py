from pydantic import BaseModel


class Issue(BaseModel):
    type: str
    line: int
    message: str


class FeatureImportance(BaseModel):
    number_of_functions:     float
    average_function_length: float
    max_nesting_depth:       float


class AnalyzeResponse(BaseModel):
    issues:             list[Issue]
    score:              float
    grade:              str
    feature_importance: FeatureImportance