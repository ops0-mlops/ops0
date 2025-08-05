import ops0
from ops0.core.graph import PipelineGraph


# Option 1 : Utiliser PipelineGraph directement (recommandé pour les tests)
def test_with_pipeline_graph():
    # Créer un pipeline avec PipelineGraph
    with PipelineGraph('test-pipeline') as pipeline:
        @ops0.step
        def test():
            return 'ok'

        # Important : appeler la fonction pour l'enregistrer
        test()

    # Exécuter
    result = ops0.run(pipeline)
    print(f'✅ Run result: {result.success}')

    # Déployer
    if result.success:
        deployment = ops0.deploy(pipeline, target='local')
        print(f'✅ Deploy local: {deployment}')

    return pipeline, result


# Option 2 : Utiliser le décorateur @ops0.pipeline
@ops0.pipeline("test-pipeline-decorated")
def test_with_decorator():
    @ops0.step
    def load_data():
        return [1, 2, 3]

    @ops0.step
    def process_data():
        data = load_data()  # Ou utiliser ops0.storage
        return [x * 2 for x in data]

    # Appeler les fonctions pour les enregistrer
    load_data()
    process_data()

    return "Pipeline créé"


# Tester les deux options
if __name__ == "__main__":
    print("=== Test avec PipelineGraph ===")
    pipeline1, result1 = test_with_pipeline_graph()

    print("\n=== Test avec décorateur ===")
    # Exécuter la fonction décorée pour créer le pipeline
    test_with_decorator()

    # Pour le décorateur, vous devez récupérer le pipeline du contexte
    current_pipeline = PipelineGraph.get_current()
    if current_pipeline:import ops0
from ops0.core.graph import PipelineGraph, StepNode
from ops0.core.analyzer import FunctionAnalyzer

# Créer un pipeline simple
pipeline = PipelineGraph('test-pipeline')

# Définir une fonction step
def test_step():
    print("Executing test step...")
    return 'ok'

# Créer les composants nécessaires
analyzer = FunctionAnalyzer(test_step)
step_node = StepNode(
    name="test_step",
    func=test_step,
    analyzer=analyzer
)

# Ajouter le step au pipeline
pipeline.add_step(step_node)

# Vérifier
print(f"Pipeline '{pipeline.name}' has {len(pipeline.steps)} steps")

# Exécuter
try:
    result = ops0.run(pipeline)
    print(f'✅ Run result: {result}')
except Exception as e:
    print(f'❌ Run error: {e}')
    import traceback
    traceback.print_exc()

# Déployer
try:
    deployment = ops0.deploy(pipeline, target='local')
    print(f'✅ Deploy result: {deployment}')
except Exception as e:
    print(f'❌ Deploy error: {e}')
    result2 = ops0.run(current_pipeline)
    print(f'✅ Decorated pipeline result: {result2.success}')