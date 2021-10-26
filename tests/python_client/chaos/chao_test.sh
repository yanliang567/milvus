set -e
set -x


echo "check os env"
platform='unknown'
unamestr=$(uname)
if [[ "$unamestr" == 'Linux' ]]; then
   platform='Linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='Mac'
fi
echo "platform: $platform"

# define chaos testing object
release=${1:-"milvus-chaos"}
pod="pulsar"
chaos_type="pod_kill"
release="milvus-chaos"
ns="chaos-testing"

# install milvus cluster for chaos testing
pushd ./scripts
echo "uninstall milvus if exist"
bash uninstall_milvus.sh ${release}|| true
echo "install milvus"
bash install_milvus.sh ${release}
popd

# replace chaos object as defined
if [ "$platform" == "Mac" ];
then
    sed -i "" "s/TESTS_CONFIG_LOCATION =.*/TESTS_CONFIG_LOCATION = \'chaos_objects\/${chaos_type}\/'/g" constants.py
    sed -i "" "s/ALL_CHAOS_YAMLS =.*/ALL_CHAOS_YAMLS = \'chaos_${pod}_podkill.yaml\'/g" constants.py
else
    sed -i "s/TESTS_CONFIG_LOCATION =.*/TESTS_CONFIG_LOCATION = \'chaos_objects\/${chaos_type}\/'/g" constants.py
    sed -i "s/ALL_CHAOS_YAMLS =.*/ALL_CHAOS_YAMLS = \'chaos_${pod}_podkill.yaml\'/g" constants.py
fi

# run chaos testing
echo "start running testcase ${pod}"
host=$(kubectl get svc/milvus-chaos -o jsonpath="{.spec.clusterIP}")
python scripts/hello_milvus.py --host "$host"
pytest -s -v test_chaos.py --host "$host" || echo "chaos test fail"
sleep 30s
echo "start running e2e test"
python scripts/hello_milvus.py --host "$host" || echo "e2e test fail"