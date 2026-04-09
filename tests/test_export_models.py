from casecrawler.models.export import Action, Conversation, Episode, Observation, TimeStep, Turn

def test_observation():
    obs = Observation(narrative="Patient presents with headache", vitals={"hr": 92, "bp_systolic": 168}, new_lab_results=[], new_imaging_results=[], pending_orders=["CBC"], time_elapsed="T+0")
    assert obs.narrative == "Patient presents with headache"
    assert obs.pending_orders == ["CBC"]

def test_action():
    a = Action(id="a1", description="Order CT head", quality="optimal")
    assert a.quality == "optimal"

def test_timestep():
    ts = TimeStep(step_number=1, observation=Observation(narrative="Presentation", vitals=None, new_lab_results=[], new_imaging_results=[], pending_orders=[], time_elapsed="T+0"), action_space=[Action(id="a1", description="Order CT head", quality="optimal"), Action(id="a2", description="Discharge", quality="catastrophic")], optimal_action="a1", reward_table={"a1": 1.0, "a2": -1.0})
    assert ts.reward_table["a1"] == 1.0

def test_episode():
    ep = Episode(case_id="test-1", difficulty="resident", specialty=["neurosurgery"], steps=[TimeStep(step_number=1, observation=Observation(narrative="Presentation", vitals=None, new_lab_results=[], new_imaging_results=[], pending_orders=[], time_elapsed="T+0"), action_space=[Action(id="a1", description="CT head", quality="optimal")], optimal_action="a1", reward_table={"a1": 1.0})])
    assert len(ep.steps) == 1

def test_conversation():
    conv = Conversation(case_id="test-1", system_prompt="You are a physician.", turns=[Turn(role="system", content="Patient presents with headache."), Turn(role="assistant", content="I would order a CT head.")])
    assert len(conv.turns) == 2
    assert conv.turns[0].role == "system"
