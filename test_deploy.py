import ops0
from ops0.core.graph import PipelineGraph

# Créer un pipeline
with PipelineGraph('test-pipeline') as pipeline:
    @ops0.step
    def test():
        return 'ok'

# Exécuter
result = ops0.run(pipeline)
print(f'✅ Run result: {result.success}')

# Déployer
if result.success:
    deployment = ops0.deploy(pipeline, target='local')
    print(f'✅ Deploy local: {deployment}')