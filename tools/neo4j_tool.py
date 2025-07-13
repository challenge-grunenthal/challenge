import os

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase


class Neo4jTool:
    def __init__(self):
        # Neo4j connection parameters
        self.URI = os.getenv("NEO4J_URI")
        self.AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        self.driver = None
        self.graph = None
        self.chain = None

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.URI, auth=self.AUTH)
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j!")

            # Initialize LangChain Neo4j graph
            self.graph = Neo4jGraph(
                url=self.URI, username=self.AUTH[0], password=self.AUTH[1]
            )

            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def get_schema_info(self):
        """Get database schema information to understand available properties"""
        if not self.driver:
            print("❌ Driver not initialized. Call connect() first.")
            return None
        
        try:
            with self.driver.session() as session:
                # Get node labels
                labels_result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = [record["label"] for record in labels_result]
                
                # Get relationship types
                rels_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rels = [record["relationshipType"] for record in rels_result]
                
                # Get properties for Drug nodes
                if "Drug" in labels:
                    drug_props_result = session.run("""
                        MATCH (d:Drug) 
                        RETURN keys(d) as properties 
                        LIMIT 1
                    """)
                    drug_props = list(drug_props_result)[0]["properties"] if drug_props_result.peek() else []
                else:
                    drug_props = []
                
                return {
                    "labels": labels,
                    "relationship_types": rels,
                    "drug_properties": drug_props
                }
        except Exception as e:
            print(f"❌ Error getting schema info: {e}")
            return None

    def initialize_qa_chain(self, openai_api_key=None):
        """Initialize the GraphCypherQAChain for natural language queries"""
        if not self.graph:
            print("❌ Graph not initialized. Call connect() first.")
            return False

        try:
            # Get OpenAI API key from environment or parameter
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print(
                    "❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass as parameter."
                )
                return False

            # Initialize the language model
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=api_key)

            # Initialize the QA chain
            self.chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=self.graph,
                verbose=True,
                return_direct=False,
                allow_dangerous_requests=True,
                top_k=5,
            )

            print("🤖 GraphCypherQAChain initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Error initializing QA chain: {e}")
            return False

    def ask_question(self, question):
        """Ask a natural language question about the graph"""
        if not self.chain:
            print("❌ QA chain not initialized. Call initialize_qa_chain() first.")
            return "Error: QA chain not initialized"

        try:
            result = self.chain.invoke({"query": question})
            return result
        except Exception as e:
            print(f"❌ Error asking question: {e}")
            return f"Error: {str(e)}"

    def get_therapeutic_categories_for_drug(self, drug_name):
        """Get therapeutic categories for drugs containing a specific substance"""
        if not self.driver:
            print("❌ Driver not initialized. Call connect() first.")
            return "Error: Driver not initialized"
        
        try:
            with self.driver.session() as session:
                # First, let's get the schema to understand available properties
                schema_info = self.get_schema_info()
                if not schema_info:
                    return "Error: Could not retrieve database schema"
                
                print(f"Available Drug properties: {schema_info['drug_properties']}")
                
                # Try different approaches based on available properties
                if "name" in schema_info['drug_properties']:
                    results = []
                    
                    # Query 1: Find drugs by name containing the substance
                    try:
                        result1 = session.run("""
                            MATCH (d:Drug)
                            WHERE toLower(d.name) CONTAINS toLower($drug_name)
                            RETURN DISTINCT d.name as drug_name, d.category as category, d.type as type
                            LIMIT 20
                        """, {"drug_name": drug_name})
                        records1 = list(result1)
                        if records1:
                            results.append(f"Query 1 results: {records1}")
                    except Exception as e:
                        results.append(f"Query 1 failed: {str(e)}")
                    
                    # Query 2: Look for therapeutic information in any available property
                    try:
                        result2 = session.run("""
                            MATCH (d:Drug)
                            WHERE toLower(d.name) CONTAINS toLower($drug_name)
                            RETURN DISTINCT d.name as drug_name, 
                                   [prop in keys(d) WHERE prop CONTAINS 'category' OR prop CONTAINS 'therapeutic' OR prop CONTAINS 'type' | prop] as relevant_properties
                            LIMIT 10
                        """, {"drug_name": drug_name})
                        records2 = list(result2)
                        if records2:
                            results.append(f"Query 2 results: {records2}")
                    except Exception as e:
                        results.append(f"Query 2 failed: {str(e)}")
                    
                    # Query 3: Find relationships to therapeutic categories if they exist
                    try:
                        result3 = session.run("""
                            MATCH (d:Drug)-[r]-(related)
                            WHERE toLower(d.name) CONTAINS toLower($drug_name)
                            AND (related:Category OR related:TherapeuticCategory OR related:Type)
                            RETURN DISTINCT d.name as drug_name, type(r) as relationship_type, related.name as category_name
                            LIMIT 20
                        """, {"drug_name": drug_name})
                        records3 = list(result3)
                        if records3:
                            results.append(f"Query 3 results: {records3}")
                    except Exception as e:
                        results.append(f"Query 3 failed: {str(e)}")
                    
                    if results:
                        return f"Found information for drugs containing '{drug_name}':\n" + "\n".join(results)
                    else:
                        return f"No drugs found containing '{drug_name}' or no therapeutic category information available."
                
                else:
                    return f"Database schema doesn't contain expected properties. Available properties: {schema_info['drug_properties']}"
                    
        except Exception as e:
            print(f"❌ Error in therapeutic categories query: {e}")
            return f"Error: {str(e)}"

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed.")
