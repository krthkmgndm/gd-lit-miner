import networkx as nx


class AssociationExtractor:
    def __init__(self, nlp_model):
        self.nlp_model = nlp_model
        self.association_graph = nx.Graph()

    def extract_associations(self, sentences):
        assocations = []

        for sentence in sentences:
            entities = self.nlp_model.identify_entities(sentence)

            for gene in entities["genes"]:
                for disease in entities["diseases"]:
                    assocation = {
                        "gene": gene,
                        "disease": disease,
                        "sentence": sentence,
                        "confidence": self._calculate_confidence(
                            sentence, gene, disease
                        ),
                    }
                    assocations.append(assocation)

        return assocations

    def _calculate_confidence(self, sentnece, gene, disease):
        # Add confidence calculation logic here
        return 0.5

    def get_network_analysis(self):
        analysis = {
            "total_associations": self.association_graph.number_of_edges(),
            "unique_genes": len(
                [
                    n
                    for n in self.association_graph.nodes()
                    if n
                    in set(
                        nx.get_node_attributes(self.association_graph, "type").keys()
                    )
                ]
            ),
            "unique_diseases": len(
                [
                    n
                    for n in self.association_graph.nodes()
                    if n
                    not in set(
                        nx.get_node_attributes(self.association_graph, "type").keys()
                    )
                ]
            ),
        }
        return analysis
