from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'go2_locomotion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'unitree_sdk2py', 'numpy', 'onnxruntime'],
    zip_safe=True,
    maintainer='csh',
    maintainer_email='wjdtmdghkss@gmail.com',
    description='Unitree Go2 locomotion control: /cmd_vel to SDK or NN policy',
    license='MIT',
    entry_points={
        'console_scripts': [
            'locomotion_node = go2_locomotion.locomotion_node:main',
        ],
    },
)
