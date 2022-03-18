pipeline {
    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')   // timeout on this stage
    }
    agent {
        kubernetes {
            label "milvus-chaos-test"
            defaultContainer 'main'
            yamlFile "build/ci/jenkins/pod/chaos-test.yaml"
            customWorkspace '/home/jenkins/agent/workspace'
            // idle 5 minutes to wait clean up tasks
            idleMinutes 5
        }
    }
    parameters{
        choice(
            description: 'Chaos Test Type',
            name: 'chaos_type',
            choices: ['pod-kill', 'pod-failure', 'mem-stress', 'network-latency', 'network-partition', 'io-latency']
        )
        choice(
            description: 'Chaos Test Target: \
            mem-stress: datanode, etcd, indexnode, minio, proxy, pulsar, querynode, standalone \
            io-fault & io-latency: minio, pulsar, etcd ',
            name: 'pod_name',
            choices: ["standalone", "datacoord", "datanode", "indexcoord", "indexnode", "proxy", "pulsar", "querycoord", "querynode", "rootcoord", "etcd", "minio"]
        )
        choice(
            description: 'Chaos Test Task',
            name: 'chaos_task',
            choices: ['chaos-test', 'data-consist-test']
        )
        string(
            description: 'Image Repository',
            name: 'image_repository',
            defaultValue: 'harbor.zilliz.cc/milvus/milvus'
        )
        string(
            description: 'Image Tag',
            name: 'image_tag',
            defaultValue: 'master-latest'
        )
        string(
            description: 'Pod Nums',
            name: 'pod_nums',
            defaultValue: '1'
        )
        booleanParam(
            description: 'Keep Env',
            name: 'keep_env',
            defaultValue: 'false'
        )
    }
    
    environment {
        ARTIFACTS = "${env.WORKSPACE}/_artifacts"
        RELEASE_NAME = "${params.pod_name}-${params.chaos_type}-${env.BUILD_ID}"
        NAMESPACE = "chaos-testing"
    }

    stages {
        stage ('Install Dependency') {
            steps {
                container('main') {
                    dir ('tests/python_client') {
                        script {
                        sh "pip install -r requirements.txt --trusted-host https://test.pypi.org"       
                        }
                    }
                }
            }
        }
        stage ('Deploy Milvus') {
            options {
              timeout(time: 10, unit: 'MINUTES')   // timeout on this stage
            }
            steps {
                container('main') {
                    dir ('tests/python_client/chaos/scripts') {
                        script {
                            def image_tag_modified = ""
                            if ("${params.image_tag}" == "master-latest") {
                                image_tag_modified = sh(returnStdout: true, script: 'bash ../../../../scripts/docker_image_find_tag.sh -n milvusdb/milvus-dev -t master-latest -f master- -F -L -q').trim()    
                            }
                            else {
                                image_tag_modified = "${params.image_tag}"
                            }
                            sh "echo ${image_tag_modified}"
                            sh "echo ${params.chaos_type}"
                            sh "docker pull ${params.image_repository}:${image_tag_modified}"
                            sh "helm repo add milvus https://milvus-io.github.io/milvus-helm"
                            sh "helm repo update"
                            if ("${params.pod_name}" == "standalone"){
                                sh"""
                                IMAGE_TAG="${image_tag_modified}" \
                                REPOSITORY="${params.image_repository}" \
                                RELEASE_NAME="${env.RELEASE_NAME}" \
                                bash install_milvus_standalone.sh
                                """    
                            }else{   
                                sh"""
                                IMAGE_TAG="${image_tag_modified}" \
                                REPOSITORY="${params.image_repository}" \
                                RELEASE_NAME="${env.RELEASE_NAME}" \
                                bash install_milvus_cluster.sh
                                """
                            }
                            sh "kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                            sh "kubectl wait --for=condition=Ready pod -l release=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                            sh "kubectl get pods|grep ${env.RELEASE_NAME}"
                            }
                        }
                    }
                }
        }
        stage ('Run e2e test before chaos') {
            options {
              timeout(time: 5, unit: 'MINUTES')   // timeout on this stage
            }            
            steps {
                container('main') {
                    dir ('tests/python_client/chaos') {
                        script {
                        def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        sh "pytest -s -v ../testcases/test_e2e.py --host $host --log-cli-level=INFO --capture=no"       
                        }
                    }
                }
            }
            
        }

        stage ('Run hello_milvus before chaos') {
            options {
              timeout(time: 5, unit: 'MINUTES')   // timeout on this stage
            }
            steps {
                container('main') {
                    dir ('tests/python_client/chaos') {
                        script {
                        def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        sh "python3 scripts/hello_milvus.py --host $host"          
                        }
                    }
                }
            }
            
        }


        stage ('Run chaos test'){
            options {
              timeout(time: 15, unit: 'MINUTES')   // timeout on this stage
            }
            steps {
                container('main') {
                    dir ('tests/python_client/chaos') {
                        script {
                        sh"""
                        POD_NAME="${params.pod_name}" \
                        CHAOS_TYPE="${params.chaos_type}" \
                        RELEASE_NAME="${env.RELEASE_NAME}" \
                        bash scripts/modify_config.sh
                        """

                        if ("${params.chaos_task}" == "chaos-test"){
                            def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            sh "pytest -s -v test_chaos.py --host $host --log-cli-level=INFO --capture=no || echo 'chaos test fail' "
                        }
                        if ("${params.chaos_task}" == "data-consist-test"){
                            def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                            sh "pytest -s -v test_chaos_data_consist.py --host $host --log-cli-level=INFO --capture=no || echo "chaos test fail" "                           
                        }
                        echo "chaos test done"
                        sh "kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"
                        sh "kubectl wait --for=condition=Ready pod -l release=${env.RELEASE_NAME} -n ${env.NAMESPACE} --timeout=360s"                               
                        sh "kubectl get pods|grep ${env.RELEASE_NAME}"
                        }
                    }
                }
            }
            
        }
        stage ('result analysis') {
            steps {
                container('main') {
                    dir ('tests/python_client/chaos/reports') {
                        script {
                            echo "result analysis"
                            sh "cat ${env.RELEASE_NAME}.log || echo 'no log file'"
                        }
                    }
                }
            }
        } 

        stage ('run e2e test after chaos') {
            options {
              timeout(time: 5, unit: 'MINUTES')   // timeout on this stage
            }            
            steps {
                container('main') {
                    dir ('tests/python_client/chaos') {
                        script {
                        def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        sh "pytest -s -v ../testcases/test_e2e.py --host $host --log-cli-level=INFO --capture=no"        
                        sh "kubectl get pods|grep ${env.RELEASE_NAME}"
                        }
                    }
                }
            }
            
        }

        stage ('Run hello_milvus after chaos') {
            options {
              timeout(time: 5, unit: 'MINUTES')   // timeout on this stage
            }
            steps {
                container('main') {
                    dir ('tests/python_client/chaos') {
                        script {
                        def host = sh(returnStdout: true, script: "kubectl get svc/${env.RELEASE_NAME}-milvus -o jsonpath=\"{.spec.clusterIP}\"").trim()
                        sh "python3 scripts/hello_milvus.py --host $host"        
                        sh "kubectl get pods|grep ${env.RELEASE_NAME}"
                        }
                    }
                }
            }
            
        }
    }
    post {
        always {
            echo 'upload logs'
            container('main') {
                dir ('tests/python_client/chaos') {
                    script {
                        echo "collecte logs"
                        sh "bash ../../scripts/export_log_k8s.sh ${env.NAMESPACE} ${env.RELEASE_NAME} k8s_log/${env.RELEASE_NAME} || true"                        
                        sh "tar -zcvf artifacts-${env.RELEASE_NAME}-pytest-logs.tar.gz /tmp/ci_logs/ --remove-files || true"
                        sh "tar -zcvf artifacts-${env.RELEASE_NAME}-server-logs.tar.gz k8s_log/ --remove-files || true"
                        archiveArtifacts artifacts: "artifacts-${env.RELEASE_NAME}-pytest-logs.tar.gz", allowEmptyArchive: true
                        archiveArtifacts artifacts: "artifacts-${env.RELEASE_NAME}-server-logs.tar.gz", allowEmptyArchive: true
                        if ("${params.keep_env}" == "false"){
                            sh "bash scripts/uninstall_milvus.sh ${env.RELEASE_NAME}"
                        }
                    }
                }
            }
        
        }
        success {
            echo 'I succeeeded!'
            container('main') {
                dir ('tests/python_client/chaos/scripts') {
                    script {
                        sh "bash uninstall_milvus.sh ${env.RELEASE_NAME} || true"
                    }
                }
            }  

        }
        unstable {
            echo 'I am unstable :/'
        }
        failure {
            echo 'I failed :('
        }
        changed {
            echo 'Things were different before...'
        }
    }
}