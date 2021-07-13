from setuptools import setup

setup(name='fairOD',
      version='0.1',
      description='Python code for FairOD: Fairness-aware Outlier Detection',
      url='https://github.com/Shubhranshu-Shekhar/fairOD',
      author='shubhranshu-shekhar',
      author_email='',
      packages=['fairod'],
      install_requires=['torch', 'numpy', 'scipy', 'scikit-learn'],
      zip_safe=False)