from distutils.core import setup

setup(name='uatu',
        version='0.1',
        description='Uatu: Probabilistic CNNs for cosmological inference.',
        author='Sean McLaughlin',
        author_email='swmclau2@stanford.edu',
        url='https://github.com/mclaughlin6464/uatu',
        packages=['uatu', 'uatu.simulations', 'uatu.watchers'])
