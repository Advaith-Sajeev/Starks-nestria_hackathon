from Detectors import models as m
import torch
from Detectors import archetecture as arc
import numpy as np


class Detector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detectors = ["model_84_acc_10_frames_final_data", "model_87_acc_20_frames_final_data",
                          "model_89_acc_40_frames_final_data", "model_90_acc_20_frames_FF_data",
                          "model_90_acc_60_frames_final_data", "model_93_acc_100_frames_celeb_FF_data",
                          "model_95_acc_40_frames_FF_data", "model_97_acc_60_frames_FF_data",
                          "model_97_acc_80_frames_FF_data", "model_97_acc_100_frames_FF_data"]
        self.processed_video_obj = m.process_video(video_path)
        self.aggregated_model = arc.CustomModel()
        self.aggregated_model.load_state_dict(torch.load('advanced_model.pth'))
        self.aggregated_model.eval()

    def individual_prob(self, selected_detectors):
        results = {}
        for d in selected_detectors:
            if d in self.detectors:
                model = m.Model(2).cuda()
                path_to_model = f'Models/{d}.pt'
                try:
                    model.load_state_dict(torch.load(path_to_model))
                    model.eval()
                    fmap, logit, weight_softmax, con = m.predict(model, self.processed_video_obj)
                    result_rgb = m.generate_image(fmap, logit, weight_softmax, self.processed_video_obj)
                    # Ensure con is a single float value
                    con = float(con)
                    results[d] = con
                except Exception as e:
                    print(f"Error loading or processing {d}: {e}")
        return results

    def aggregate(self, selected_detectors):
        # Given dictionary of accuracies
        accuracy_dict = self.individual_prob(selected_detectors)

        # Extract probabilities
        model_names = list(accuracy_dict.keys())
        model_accuracies = np.array(list(accuracy_dict.values()))
        probabilities = model_accuracies / 100.0

        # Reshape the probabilities to match the model input shape
        new_inputs = torch.FloatTensor(probabilities.reshape(1, -1))
        with torch.no_grad():
            predictions = self.aggregated_model(new_inputs)

        accuracy_dict["aggregated_probability"] = float(predictions.item())  # Convert predictions to float
        return accuracy_dict


d = Detector(r"Elon Musk Deepfake Introduces Person-Specific Deepfakes.mp4")
selected_detectors = ["model_84_acc_10_frames_final_data", "model_87_acc_20_frames_final_data",
                      "model_89_acc_40_frames_final_data", "model_90_acc_20_frames_FF_data",
                      "model_90_acc_60_frames_final_data", "model_93_acc_100_frames_celeb_FF_data",
                      "model_95_acc_40_frames_FF_data", "model_97_acc_60_frames_FF_data",
                      "model_97_acc_80_frames_FF_data", "model_97_acc_100_frames_FF_data"]
results = d.aggregate(selected_detectors)
print(results)
