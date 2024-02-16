import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from inference.models.yolo_world.yolo_world import YOLOWorld

from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class YOLOWorldModel(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.model = YOLOWorld( model_id="yolo_world/l")
        self.ontology = ontology
        self.model.set_classes(self.ontology.prompts())
        self.model.names = self.ontology.prompts()

    def predict(self, input: str, confidence: int = 0.2) -> sv.Detections:
        results = self.model.infer(load_image(input), text=self.ontology.prompts())

        detections = sv.Detections.from_inference(results)

        detections = detections[detections.confidence > confidence]

        return detections