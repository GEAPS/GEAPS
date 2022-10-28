import pickle

from mrl.algorithms.ant_skill_policy import AntSkillPolicy
from mrl.algorithms.fpp_skill_policy import FppSkillPolicy
from mrl.algorithms.pm_skill_policy import PmSkillPolicy
from mrl.algorithms.edl_skill_policy import EDLAntSkillPolicy
    
def get_skill_policy(env_name, n_latents, skill_type="maxent"):
    if skill_type != "maxent":
        assert "ant" in env_name.lower(), "%s skills not avilable for %s" % (skill_type, env_name)
    if "ant" in env_name.lower():
        if skill_type == "maxent":
            with open("mrl/algorithms/models/ant_skill_params.pkl", "rb") as f:
                policy_param_list = pickle.load(f)
            skill_policy = AntSkillPolicy(env_name, policy_param_list, "cuda", n_latents)
        elif skill_type == "snn4hrl":
            with open("mrl/algorithms/models/snn4hrl/ant_snn4hrl.pkl", "rb") as f:
                policy_param_list = pickle.load(f)
            skill_policy = AntSkillPolicy(env_name, policy_param_list, "cuda", n_latents)
        elif skill_type == "edl":
            skill_policy = EDLAntSkillPolicy(env_name, "cuda")
        else:
            raise ValueError("Unknown skill type")
    elif "pickplace" in env_name.lower() or "stack" in env_name.lower():
        skill_policy= FppSkillPolicy(n_latents)
    elif "pointmaze" in env_name.lower():
        with open("mrl/algorithms/models/pm_skill_params.pkl", "rb") as f:
            policy_param_list = pickle.load(f)
        skill_policy = PmSkillPolicy(policy_param_list, "cuda", n_latents)
    else:
        raise ValueError("The corresponding skill doesn't exist.")
  
    return skill_policy
