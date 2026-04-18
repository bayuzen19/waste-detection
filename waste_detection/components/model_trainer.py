import os, sys
import shutil
import subprocess
import yaml
from waste_detection.logger import logging
from waste_detection.exception import AppException
from waste_detection.entity.config_entity import ModelTrainerConfig
from waste_detection.entity.artifact_entity import ModelTrainerArtifact
from waste_detection.utils.main_utils import read_yaml_file

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered the initiate_model_trainer method of ModelTrainer class")

        try:
            data_yaml_path = os.path.join("artifacts", "feature_store", "data.yaml")
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"Data config file not found: {data_yaml_path}")

            with open(data_yaml_path, "r") as f:
                dataset_config = yaml.safe_load(f)

            dataset_root_dir = os.path.dirname(data_yaml_path)
            train_images_dir = os.path.abspath(os.path.join(dataset_root_dir, "train", "images"))
            valid_images_dir = os.path.abspath(os.path.join(dataset_root_dir, "valid", "images"))

            if not os.path.exists(train_images_dir) or not os.path.exists(valid_images_dir):
                raise FileNotFoundError(
                    f"Dataset image directories not found: train={train_images_dir}, val={valid_images_dir}"
                )

            dataset_config["train"] = train_images_dir
            dataset_config["val"] = valid_images_dir
            custom_data_yaml_path = "./yolov5/data/custom_data.yaml"
            with open(custom_data_yaml_path, "w") as f:
                yaml.safe_dump(dataset_config, f, sort_keys=False)

            num_classes = str(dataset_config["nc"])
            
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"./yolov5/models/{model_config_file_name}.yaml")

            config["nc"] = int(num_classes)

            with open(f"./yolov5/models/custom_{model_config_file_name}.yaml", "w") as f:
                yaml.dump(config, f)
            
            run_name = f"custom_{model_config_file_name}_results"
            train_command = [
                sys.executable,
                "train.py",
                "--img",
                "416",
                "--batch",
                str(self.model_trainer_config.batch_size),
                "--epochs",
                str(self.model_trainer_config.no_epochs),
                "--data",
                "./data/custom_data.yaml",
                "--cfg",
                f"./models/custom_{model_config_file_name}.yaml",
                "--weights",
                self.model_trainer_config.weight_name,
                "--name",
                run_name,
                "--cache",
                "ram",
            ]
            subprocess.run(train_command, cwd="./yolov5", check=True)

            best_model_path = f"./yolov5/runs/train/{run_name}/weights/best.pt"
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Trained model file not found: {best_model_path}")

            shutil.copy2(best_model_path, "./yolov5/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy2(best_model_path, self.model_trainer_config.model_trainer_dir)

            os.system("rm -rf ./yolov5/runs")

            model_trainer_artifact = ModelTrainerArtifact(training_model_file_path="yolov5/best.pt")
            logging.info("Trained the model and got the best model file path")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info("Exited the initiate_model_trainer method of ModelTrainer class")
            return model_trainer_artifact          

        except Exception as e:
            raise AppException(e, sys) from e