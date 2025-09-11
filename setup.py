from setuptools import find_packages, setup
from typing import List

Hypen_e_Dot = "-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if Hypen_e_Dot in requirements:
            requirements.remove(Hypen_e_Dot)
          
    return requirements

setup(
    name='math_score_predictor',
    version='0.0.2',
    author='Mohammed Adam',
    author_email='mail.mohammedadam@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


