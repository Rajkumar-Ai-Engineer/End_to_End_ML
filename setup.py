from setuptools import find_packages,setup 

def get_requirements(file_path:str="requirements.txt"):
    
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        requirements = requirements.remove("-e .")  if "-e ." in requirements else requirements
        return requirements




setup(
    name="Students_Mark_Prediction",
    author="Rajkumar",
    author_email="tech84602@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements(file_path="requirements.txt"),
)