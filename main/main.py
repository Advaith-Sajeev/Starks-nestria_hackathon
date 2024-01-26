from concurrent.futures import ThreadPoolExecutor
from detectors import models as m

class Detector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detectors = ["model_84_acc_10_frames_final_data", "model_87_acc_20_frames_final_data",
                          "model_89_acc_40_frames_final_data", "model_90_acc_20_frames_FF_data",
                          "model_90_acc_60_frames_final_data", "model_93_acc_100_frames_celeb_FF_data",
                          "model_95_acc_40_frames_FF_data", "model_97_acc_60_frames_FF_data",
                          "model_97_acc_80_frames_FF_data", "model_97_acc_100_frames_FF_data"]

    def process_video(self):
        pass

    def process_and_predict_parallel(self, detector):
        return detector, m.process_and_predict(self.video_path, detector)

    def aggregate(self, selected_detectors):
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_and_predict_parallel, detector) for detector in selected_detectors]

            for future in futures:
                detector, result = future.result()
                if detector in self.detectors:
                    results[detector] = result
                else:
                    print(f"Detector {detector} not found.")

        return results


d = Detector(r"C:\Users\mahes\Downloads\Elon Musk Deepfake Introduces Person-Specific Deepfakes.mp4")
selected_detectors = ["model_84_acc_10_frames_final_data", "model_87_acc_20_frames_final_data",
                      "model_89_acc_40_frames_final_data", "model_90_acc_20_frames_FF_data",
                      "model_90_acc_60_frames_final_data", "model_93_acc_100_frames_celeb_FF_data",
                      "model_95_acc_40_frames_FF_data", "model_97_acc_60_frames_FF_data",
                      "model_97_acc_80_frames_FF_data", "model_97_acc_100_frames_FF_data"]
results = d.aggregate(selected_detectors)
print(results)
