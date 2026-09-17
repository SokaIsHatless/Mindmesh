from models import CalculationRequest
from normalizer import normalize_request

req = CalculationRequest(
    status="ok",
    operation="velocity",
    inputs={"distance": 120, "time_min": 30},
)
print(normalize_request(req))
# operation='speed', inputs={'distance_km': 120.0, 'time_hr': 0.5}