from setuptools import setup
from glob import glob
import os


package_name = 'occupancy_grid'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib', 'scikit-learn'],
    zip_safe=True,
    maintainer='didikid',
    maintainer_email='didikid3@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'occupancy_grid_node = occupancy_grid.occupancy_grid_node:main',
            'simple_frontier_node = occupancy_grid.simple_frontier_node:main',
            'center_line_node = occupancy_grid.center_line_node:main'
        ],
    },
)
