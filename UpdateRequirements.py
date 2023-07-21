import os 

def updateRequirements(install=False):
    if install:
        os.system('pip install -r requirements.txt')
    else:
        os.system('pip freeze > requirements.txt')


if __name__ == '__main__':
    updateRequirements(install=False)