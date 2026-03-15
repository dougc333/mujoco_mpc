
#install for agent.py

python3 -m pip install grpcio grpcio-tools protobuf
python3 setup.py build_py
python3 -m pip install -e .



python3 setup.py build_py
cp -r build/lib.macosx-11.0-arm64-cpython-311/mujoco_mpc/proto mujoco_mpc/
python3 -m pip install -e .

