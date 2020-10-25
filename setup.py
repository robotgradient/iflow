from setuptools import setup
from codecs import open
from os import path


from iflow import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='iflow',
      version=__version__,
      description='ImitationFlow: Learning Deep Stable Stochastic Dynamic Systems by Normalizing Flows',
      author='Julen Urain',
      author_email='julen@robot-learning.de',
      packages=['iflow'],
      install_requires=requires_list,
      )
