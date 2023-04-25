from setuptools import setup

setup(
    name="nkf_aec_tools",
    version="1.0.0",
    description="Tools to perform AEC using NKF-AEC (https://github.com/fjiang9/NKF-AEC)",
    author="MayamaTakeshi",
    author_email="mayamatakeshi@gmail.com",
    packages=["nkf_aec_tools"],
    entry_points={
        "console_scripts": [
            "nkf-aec=nkf_aec_tools.nkf_aec:main",
            "dir-nkf-aec=nkf_aec_tools.dir_nkf_aec:main",
            "nkf-aec-server=nkf_aec_tools.nkf_aec_server:main",
        ]
    },
    install_requires=[
        "torch==1.9.0+cpu",
        "torchvision==0.10.0+cpu",
        "soundfile",
    ],
    include_package_data=True
)
