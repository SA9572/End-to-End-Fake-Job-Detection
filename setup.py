from setuptools import setup, find_packages

setup(
    name='fakejobdetector',
    version='0.1',
    description='Fake Job Posting Detection ML Project',
    author='Your Name',
    author_email='you@example.com',
    packages=find_packages(),  # Automatically includes all packages in src/
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'joblib',
        'flask',
        'matplotlib',
        'seaborn'
    ],
    include_package_data=True,
)
