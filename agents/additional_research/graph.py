from py2neo import Graph, Node, Relationship

graph = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))

# Clear the DB
graph.delete_all()

# Nodes
xisen = Node("Person", name="Xisen", role="Founder", email="xisen@pulse.ai")
investor = Node("Person", name="Investor A", role="VC", firm="Index Ventures", email="a@indexvc.com")

email1 = Node("Interaction", type="email", subject="Pulse Series A intro", date="2023-11-03")
meeting1 = Node("Interaction", type="meeting", date="2023-11-10", topic="Fundraising strategy")
doc_shared = Node("Document", title="Pulse Roadmap", type="Notion", shared_on="2023-11-05")

# Connection with rich attributes
connection = Relationship(xisen, "CONNECTED_TO", investor,
                          social_type="professional",
                          relationship="founder-investor",
                          strength="high",
                          since="2023-11",
                          connection_points=str([
                              "intro by James (CEO at Pastel)",
                              "email thread on fundraising",
                              "shared roadmap document",
                              "Zoom meeting on Nov 10"
                          ]))

# Edges to interactions
x_to_email = Relationship(xisen, "SENT", email1)
i_to_email = Relationship(investor, "RECEIVED", email1)
x_to_meeting = Relationship(xisen, "ATTENDED", meeting1)
i_to_meeting = Relationship(investor, "ATTENDED", meeting1)
x_to_doc = Relationship(xisen, "SHARED", doc_shared)
i_to_doc = Relationship(investor, "RECEIVED", doc_shared)

# Push to graph
graph.create(xisen | investor | email1 | meeting1 | doc_shared |
             connection | x_to_email | x_to_meeting | x_to_doc |
             i_to_email | i_to_meeting | i_to_doc)