import os 

files_list = ["./requirements.txt","./setup.py","./README.md","src/__init__.py","./exceptons.py","./logger.py","./config.py","./utils.py","./src/components/data_ingestion.py","./src/components/data_validation.py","./src/components/data_transformation.py","./src/components/model_trainer.py","./src/components/model_evaluation.py","./src/components/model_pusher.py","./src/pipeline/training_pipeline.py","./src/components/__init__.py","./src/pipeline/__init__.py"]

def create_template(files_list:list):
    for file_path in files_list:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"a") as f:
            f.close()
            


if __name__ == "__main__":
    print("template file creation started")
    create_template(files_list)
    print("template file creation completed")