from powerbi_ontology import PowerBIExtractor

extractor = PowerBIExtractor("powerbi_test.pbix")
ontology = extractor.extract().to_ontology()
ontology.export_fabric_iq("ontology_test.json")




#Or,
from powerbi_ontology import PowerBIExtractor, OntologyGenerator

# Step 1: Extract semantic model from Power BI
extractor = PowerBIExtractor("powerbi_test.pbix")
semantic_model = extractor.extract()

# Step 2: Generate formal ontology
generator = OntologyGenerator(semantic_model)
ontology = generator.generate()

print(f"✅ Extracted {len(ontology.entities)} entities")
print(f"✅ Generated {len(ontology.business_rules)} business rules")

# Step 3: Export to your preferred format
from powerbi_ontology.export import FabricIQExporter, OntoGuardExporter

fabric_exporter = FabricIQExporter(ontology)
fabric_json = fabric_exporter.export()

ontoguard_exporter = OntoGuardExporter(ontology)
ontoguard_json = ontoguard_exporter.export()

import json

# Convert to a JSON string (with indent for readability)
ontoguard_json_string = json.dumps(ontoguard_json, indent=4, ensure_ascii=False)

# Print it or save it
print(ontoguard_json_string)

# To save it to a file:
with open('ontoguard_export.json', 'w', encoding='utf-8') as f:
    json.dump(ontoguard_json, f, indent=4, ensure_ascii=False)
