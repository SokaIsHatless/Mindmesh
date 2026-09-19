from models import ToolSpec
import test_generator

spec = ToolSpec(
    name="speed",
    purpose="compute speed from distance and time",
    operation="speed",
    inputs=[
        {
            "name": "distance_km",
            "type": "float",
            "unit": "km",
            "required": True
        },
        {
            "name": "time_hr",
            "type": "float",
            "unit": "hr",
            "required": True
        }
    ],
    output={
        "name": "speed_kmh",
        "type": "number",
        "unit": "km/h"
    },
    constraints=[
        "time_hr must not be zero"
    ],
    examples=[
        {
            "inputs": {
                "distance_km": 120,
                "time_hr": 2
            },
            "expected": 60
        },
        {
            "inputs": {
                "distance_km": 90,
                "time_hr": 3
            },
            "expected": 30
        }
    ],
    allowed_dependencies=[]
)

result = test_generator.generate_tests(spec)

print(result.model_dump_json(indent=2))