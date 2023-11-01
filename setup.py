from setuptools import setup, find_packages

setup(
    name='snap',
    version='0.1',
    author='Abdulkadir Canatar, Jenelle Feather',
    author_email='canatara@gmail.com, jfeather@flatironinstitute.org',
    packages=find_packages(include=['snap', 'snap.*']),
    install_requires=[
        'jupyter',
        'scikit-learn',
        'gitpython',
        'peewee',
        'fire',
        'psycopg2-binary',
        'pybtex'
    ],
)

# !pip install git+https://github.com/brain-score/brainio.git
# !pip install git+https://github.com/brain-score/result_caching
# !pip install --no-deps git+https://github.com/brain-score/brain-score
