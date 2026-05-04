from ..utilities import core_method
from ..utilities import helper_method
from .model_inspector import ModelInspector

class ModelSummary:
    
    @core_method
    @classmethod
    def build(cls, model):
        breakdown = ModelInspector.parameters_breakdown(model)
        return breakdown
    
    @helper_method
    def log(cls, model):
        breakdown = cls.build(model)
        print("\nNeural Network Summary\n")
        print(f"Number of layers: {model.__L}")
        print(f"Hidden Activation : {model.__hidden_activation_type.name}")
        print(f"Output Activation : {model.__output_activation_type.name}")
        print(f"Loss Type : {model.__loss_type.name}")
        print(f"Total Number of Parameters : {ModelInspector.count_parameters(model)}")
        print(f"Breakdown : {breakdown}")