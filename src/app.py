from RAG import RetrievalDoc, BuildKnowledgeGraph, read_config, KnowledgeGraphQuery
from flask import Flask, request, jsonify
import json
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize RAG and create index
config = read_config()
RAG = RetrievalDoc(config=config)
RAG.create_index()  # Create index once at startup
KG=BuildKnowledgeGraph(config=config)
KG.prepare_data()
KG.generate_knowledge_graph_llm(config=config)

@app.route('/predict', methods=['POST'])
def predict():
    while True:
        """Use this question: 
        
        Patient Information:
        Jane Doe, a 58-year-old female, was admitted on June 15, 2024.
        Chief Complaint and History of Present Illness:
        Jane reported a high fever up to 104°F, body pain, and a rash, 
        starting five days prior to admission.

        Past Medical History:
        Jane has no significant past medical history and no known allergies.

        Physical Examination:
        Jane's temperature was 102.8°F, heart rate 110 bpm, blood pressure 100/70 mmHg, and respiratory rate 20 breaths 
        per minute. No petechiae or purpura were noted."""
        
        question = input("Ask a question:")
        # Get RAG answer
        answer = RAG.get_results(question)
        KGQ = KnowledgeGraphQuery(config=config)
        result = KGQ.get_query(config=config)

        return jsonify({'kg_result': result})

if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0', debug=True)
