import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from ultralytics import YOLOWorld as YOLOWorldModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class YOLOWorld(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.model = YOLOWorldModel("yolo-world.pt")
        self.ontology = ontology
        self.model.set_classes(self.ontology.prompts())
        self.model.model.names = self.ontology.prompts()

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        results = self.model.predict(load_image(input))

        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.confidence > confidence]

        return detections