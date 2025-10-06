# 🛫 **Introduction to Embodied Intelligence (A Quick Guide of Embodied-AI)**

# 🤖 **About**

With the rapid advancement of large-scale models, a key question that has drawn considerable attention among researchers is how to equip a model-based intelligent agent with a physical body capable of interacting with the real world. In response, the concept of embodied intelligence has been introduced, attracting a growing community of researchers. To help researchers quickly grasp the frontiers of embodied intelligence and intelligent robotics—and to better promote and publicize developments in this field—this project summarizes representative works in the domain of embodied intelligence and intelligent robotics. It will be continually updated to remain at the cutting edge. If you find any errors while reading through this project, please do not hesitate to contact us to correct them; we greatly appreciate such feedback. Likewise, if you would like to contribute to the further exploration and promotion of embodied intelligence, you are welcome to reach out to me via email: <yinchenghust@outlook.com>. BTW, please feel free to cite or open pull requests.
![cover](Assets/cover.png)

# Author List

Thank you to all the authors for their contributions to the project.

Cheng Yin, Chenyu Yang, Zhiwen Hu, Yunxiang Mi, Weichen Lin, Yimeng Wang.

# **Table of Contents**

- [🛫 **Introduction to Embodied Intelligence (A Quick Guide of Embodied-AI)**](#-introduction-to-embodied-intelligence-a-quick-guide-of-embodied-ai)
- [🤖 **About**](#-about)
- [Author List](#author-list)
- [**Table of Contents**](#table-of-contents)
- [**Symbol representation**](#symbol-representation)
- [📑 **Survey**](#-survey)
- [👁️ **Perception**](#️-perception)
- [🧠 **Brain Model**](#-brain-model)
- [🏆 **VLA Model**](#-vla-model)
- [👑 **E-AI-RL**](#-e-ai-rl)
- [🏁 **Interactive**](#-interactive)
- [💻 **Simulator**](#-simulator)
- [📊 **Dataset**](#-dataset)
- [🔧 **Toolkit**](#-toolkit)
- [📖 **Citation**](#-citation)
- [😊 **Acknowledgements**](#-acknowledgements)

# **Symbol representation**

- [ ] represents closed source.
- [x] represents open source.

# 📑 **Survey**

- Teleoperation of Humanoid Robots: A Survey [[Paper Link]](https://arxiv.org/pdf/2301.04317) [[Project Link]](https://humanoid-teleoperation.github.io) [2023]
- Deep Learning Approaches to Grasp Synthesis: A Review [[Paper Link]](https://arxiv.org/pdf/2301.04317) [[Project Link]](https://rhys-newbury.github.io/projects/6dof/) [2023]
- A survey of embodied ai: From simulators to research tasks [[Paper Link]](https://arxiv.org/pdf/2103.04918) [2022]
- A Survey of Embodied Learning for Object-Centric Robotic Manipulation [[Paper Link]](https://arxiv.org/pdf/2408.11537) [[Project Link]](https://github.com/RayYoh/OCRM_survey) [2024]
- A Survey on Vision-Language-Action Models for Embodied AI [[Paper Link]](https://arxiv.org/abs/2405.14093) [2024]
- Embodied Intelligence Toward Future Smart Manufacturing in the Era of AI Foundation Model [[Paper Link]](https://ieeexplore.ieee.org/document/10697107) [2024]
- Towards Generalist Robot Learning from Internet Video: A Survey [[Paper Link]](https://arxiv.org/abs/2404.19664) [2024]
- A Survey on Robotics with Foundation Models: toward Embodied AI [[Paper Link]](https://arxiv.org/abs/2402.02385) [2024]
- Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis [[Paper Link]](https://arxiv.org/abs/2312.08782)[[Project Link]](https://github.com/JeffreyYH/Awesome-Generalist-Robots-via-Foundation-Models) [2024]
- Robot Learning in the Era of Foundation Models: A Survey [[Paper Link]](https://arxiv.org/abs/2311.14379) [2023]
- Foundation Models in Robotics: Applications, Challenges, and the Future [[Paper Link]](https://arxiv.org/abs/2312.07843) [[Project Link]](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models) [2023]
- Large Language Models for Robotics: Opportunities, Challenges, and Perspectives [[Paper Link]](https://arxiv.org/abs/2401.04334) [2024]
- Awesome-Embodied-Agent-with-LLMs [[Project Link]](https://github.com/zchoi/Awesome-Embodied-Agent-with-LLMs) [2024]
- Awesome Embodied Vision [[Project Link]](https://github.com/ChanganVR/awesome-embodied-vision) [2024]
- Awesome Touch [[Project Link]](https://github.com/linchangyi1/Awesome-Touch) [2024]
- Grasp-Anything Project [[Project Link]](https://airvlab.github.io/grasp-anything/) [2024]
- GraspNet Project [[Project Link]](https://graspnet.net/anygrasp) [2024]
- Deep Generative Models for Offline Policy Learning:
  Tutorial, Survey, and Perspectives on Future Directions [[Paper Link]](https://arxiv.org/pdf/2402.13777) [[Project Link]](https://github.com/LucasCJYSDL/DGMs-for-Offline-Policy-Learning) [2024]
- Survey of Learning-based Approaches for Robotic
  In-Hand Manipulation [[Paper Link]](https://arxiv.org/pdf/2401.07915) [2024]
- A Survey of Optimization-based Task and Motion Planning: From Classical To Learning Approaches [[Paper Link]](https://arxiv.org/abs/2404.02817) [2024]
- Neural Scaling Laws in Robotics [[Paper Link]](https://arxiv.org/abs/2405.14005) [2025]
- Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes [[Paper Link]](https://arxiv.org/abs/2408.03539) [2024]
- Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI [[Paper Link]](https://arxiv.org/abs/2407.06886) [[Project Link]](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List) [2024]
- Controllable Text Generation for Large Language Models: A Survey [[Paper Link]](https://arxiv.org/abs/2408.12599) [[Project Link]](https://github.com/IAAR-Shanghai/CTGSurvey) [2024]
- Bridging Language and Action A Survey of Language-Conditioned Robot Manipulation [[Paper Link]](https://arxiv.org/abs/2312.10807) [2023]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 👁️ **Perception**

- - [ ] RGBGrasp: Image-based Object Grasping by Capturing Multiple Views during Robot Arm Movement with Neural Radiance Fields [[Paper Link]](https://arxiv.org/pdf/2311.16592) [[Project Link]](https://sites.google.com/view/rgbgrasp) [2024]
- - [x] RGBManip: Monocular Image-based Robotic Manipulation through Active Object Pose Estimation [[Paper Link]](https://arxiv.org/pdf/2310.03478) [[Project Link]](https://github.com/hyperplane-lab/RGBManip) [2024]
- - [x] ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2312.16217) [[Project Link]](https://github.com/clorislili/ManipLLM) [2023]
- - [x] Play to the Score: Stage-Guided Dynamic Multi-Sensory Fusion for Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2408.01366) [[Project Link]](https://gewu-lab.github.io/MS-Bot/) [2024]
- - [ ] A Contact Model based on Denoising Diffusion to Learn Variable Impedance Control for Contact-rich Manipulation [[Paper Link]](https://arxiv.org/abs/2403.13221) [2024]
- - [x] FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html) [[Project Link]](https://github.com/NVlabs/FoundationPose) [2024]
- - [x] BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.html) [[Project Link]](https://github.com/NVlabs/BundleSDF) [2023]
- - [x] TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation [[Paper Link]](https://arxiv.org/abs/2409.11047) [[Project Link]](https://github.com/popnut123/TacDiffusion) [2024]
- - [x] GelFusion: Enhancing Robotic Manipulation under Visual Constraints via Visuotactile Fusion [[Paper Link]](https://arxiv.org/abs/2505.07455) [[Project Link]](https://github.com/GelFusion/GelFusion) [2025]
- - [x] Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/9340777) [[Project Link]](https://github.com/skumra/robotic-grasping) [2020]
- - [x] Touch begins where vision ends: Generalizable policies for contact-rich manipulation [[Paper Link]](https://arxiv.org/abs/2506.13762) [[Project Link]](https://github.com/Exiam6/ViTaL) [2025]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 🧠 **Brain Model**

- - [x] RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning [[Paper Link]](https://arxiv.org/abs/2409.14674) [[Project Link]](https://github.com/rich-language-failure-recovery/Open-LLaVA-NeXT/tree/racer_llava?tab=readme-ov-file#51-set-up-language-encoder-service-ie-clip-and-t5-model-around-20gb-in-total) [2024]
- - [x] Errors are Useful Prompts: Instruction Guided Task Programming with Verifier-Assisted Iterative Prompting [[Paper Link]](https://arxiv.org/abs/2303.14100) [[Project Link]](https://github.com/ac-rad/xdl-generation) [2023]
- - [x] Generalized Planning in PDDL Domains with Pretrained Large Language Models [[Paper Link]](https://arxiv.org/pdf/2305.11014) [[Project Link]](https://github.com/tomsilver/llm-genplan/) [2023]
- - [x] QueST: Self-Supervised Skill Abstractions for Learning Continuous Control [[Paper Link]](https://arxiv.org/pdf/2407.15840) [[Project Link]](https://github.com/pairlab/QueST) [2024]
- - [ ] Plan Diffuser: Grounding LLM Planners with Diffusion Models for Robotic Manipulation [[Paper Link]](https://openreview.net/pdf?id=2a3sgm5YeX) [2024]
- - [ ] Action-Free Reasoning for Policy Generalization [[Paper Link]](https://arxiv.org/pdf/2502.03729) [[Project Link]](https://rad-generalization.github.io) [2025]
- - [ ] Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection [[Paper Link]](https://arxiv.org/abs/2412.04455) [[Project Link]](https://zhoues.github.io/Code-as-Monitor/) [2024]
- - [ ] DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment [[Paper Link]](https://arxiv.org/abs/2307.00329) [[Project Link]](https://sites.google.com/view/doremi-paper)
- - [x] Chain-of-Thought Predictive Control [[Paper Link]](https://arxiv.org/abs/2304.00776) [[Project Link]](https://sites.google.com/view/cotpc) [2024]
- - [x] CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2411.19650) [[Project Link]](https://cogact.github.io) [2024]
- - [x] ClevrSkills: Compositional Language and Visual Reasoning in Robotics [[Paper Link]](https://arxiv.org/abs/2411.09052) [[Project Link]](https://github.com/Qualcomm-AI-research/ClevrSkills) [2024]
- - [x] RoboMatrix: A Skill-centric Hierarchical Framework for Scalable Robot Task Planning and Execution in Open-World [[Paper Link]](https://arxiv.org/abs/2412.00171) [[Project Link]](https://github.com/WayneMao/RoboMatrix) [2024]
- - [ ] Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning [[Paper Link]](https://arxiv.org/abs/2311.17842) [[Project Link]](https://robot-vila.github.io) [2023]
- - [x] Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation [[Paper Link]](https://arxiv.org/abs/2401.02117) [[Project Link]](https://mobile-aloha.github.io/cn.html) [2024]
- - [x] DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation [[Paper Link]](https://arxiv.org/abs/2403.07788) [[Project Link]](https://github.com/j96w/DexCap) [2024]
- - [x] HumanPlus: Humanoid Shadowing and Imitation from Humans [[Paper Link]](https://arxiv.org/abs/2406.10454) [[Project Link]](https://humanoid-ai.github.io/) [2024]
- - [x] On Bringing Robots Home [[Paper Link]](https://arxiv.org/abs/2311.16098) [[Project Link]](https://github.com/notmahi/dobb-e) [2023]
- - [x] Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots [[Paper Link]](https://arxiv.org/abs/2402.10329) [[Project Link]](https://umi-gripper.github.io/) [2024]
- - [x] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion [[Paper Link]](https://arxiv.org/abs/2303.04137v4) [[Project Link]](https://diffusion-policy.cs.columbia.edu/) [2023]
- - [x] Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware [[Paper Link]](https://arxiv.org/abs/2304.13705) [[Project Link]](https://arxiv.org/abs/2304.13705) [2023]
- - [x] Surgical Robot Transformer (SRT): Imitation Learning for Surgical Tasks [[Paper Link]](https://arxiv.org/abs/2407.12998) [[Project Link]](https://surgical-robot-transformer.github.io/) [2024]
- - [x] Large Language Models for Orchestrating Bimanual Robots [[Paper Link]](https://arxiv.org/abs/2404.02018)  [[Project Link]](https://labor-agent.github.io/)  [2024]
- - [x] Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[Paper Link]](https://arxiv.org/abs/2204.01691)  [[Project Link]](https://say-can.github.io)  [2022]
- - [x] ManipLLM: Embodied MLLM for Object-Centric Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2312.16217)  [[Project Link]](https://github.com/clorislili/ManipLLM)  [2023]
- - [x] 3D Diffusion Policy:Generalizable Visuomotor Policy Learning via Simple 3D Representations [[Paper Link]](https://arxiv.org/abs/2403.03954)  [[Project Link]](https://github.com/YanjieZe/3D-Diffusion-Policy)  [2024]
- - [x] Prediction with Action: Visual Policy Learning via Joint Denoising Process [[Paper Link]](https://arxiv.org/abs/2411.18179)  [[Project Link]](https://github.com/Robert-gyj/Prediction_with_Action)  [2024]
- - [x] Real-World Humanoid Locomotion with Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/2303.03381)  [[Project Link]](https://github.com/kyegomez/HLT)  [2023]
- - [ ] Humanoid Locomotion as Next Token Prediction [[Paper Link]](https://arxiv.org/abs/2402.19469)  [2024]
- - [x] OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement [[Paper Link]](https://arxiv.org/abs/2503.17352)  [[Project Link]](https://github.com/yihedeng9/OpenVLThinker)  [2025]
- - [x] LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL [[Paper Link]](https://arxiv.org/abs/2503.07536)  [[Project Link]](https://github.com/TideDra/lmm-r1)  [2025]
- - [ ] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/2503.08751)  [2025]
- - [ ] VTAO-BiManip: Masked Visual-Tactile-Action Pre-training with Object Understanding for Bimanual Dexterous Manipulation [[Paper Link]](https://arxiv.org/abs/2501.03606)  [2025]
- - [ ] Real-World Humanoid Locomotion with Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/2303.03381)  [2023]
- - [x] LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL [[Paper Link]](https://arxiv.org/abs/2410.21845)  [[Project Link]](https://github.com/rail-berkeley/hil-serl)  [2025]
- - [x] Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation [[Paper Link]](https://arxiv.org/abs/2403.04436)  [[Project Link]](https://github.com/LeCAR-Lab/human2humanoid)  [2024]
- - [ ] On the Modeling Capabilities of Large Language Models for Sequential Decision Making [[Paper Link]](https://arxiv.org/abs/2410.05656)  [2024]
- - [x] RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback [[Paper Link]](https://arxiv.org/abs/2402.03681)  [[Project Link]](https://github.com/yufeiwang63/RL-VLM-F)  [2024]
- - [ ] Embodied Task Planning with Large Language Models [[Paper Link]](https://arxiv.org/abs/2307.01848)  [2023]
- - [ ] RoboGPT: an intelligent agent of making embodied long-term decisions for daily instruction tasks [[Paper Link]](https://arxiv.org/abs/2311.15649)  [2023]
- - [x] SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks [[Paper Link]](https://arxiv.org/abs/2305.17390)  [[Project Link]](https://swiftsage.github.io)  [2023]
- - [ ] An Interactive Agent Foundation Model [[Paper Link]](https://arxiv.org/abs/2402.05929)  [2024]
- - [x] Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models [[Paper Link]](https://arxiv.org/abs/2304.09842)  [[Project Link]](https://chameleon-llm.github.io)  [2023]
- - [x] Octopus: Embodied Vision-Language Programmer from Environmental Feedback [[Paper Link]](https://arxiv.org/abs/2310.08588)  [[Project Link]](https://choiszt.github.io/Octopus/)  [2023]
- - [ ] RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control [[Paper Link]](https://arxiv.org/abs/2307.15818)  [2023]
- - [x] OpenVLA: An Open-Source Vision-Language-Action Model [[Paper Link]](https://arxiv.org/abs/2406.09246)  [[Project Link]](https://openvla.github.io)  [2024]
- - [x] π0: A Vision-Language-Action Flow Model for General Robot Control [[Paper Link]](https://arxiv.org/abs/2410.24164)  [[Project Link]](https://github.com/Physical-Intelligence/openpi)  [2024]
- - [x] Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks[[Paper Link]](https://arxiv.org/abs/2503.21696)  [[Project Link]](https://github.com/zwq2018/embodied_reasoner)  [2025]
- - [ ] DoorBot: Closed-Loop Task Planning and Manipulation for Door Opening in the Wild with Haptic Feedback [[Paper Link]](https://arxiv.org/abs/2504.09358) [2025]
- - [x] IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models [[Paper Link]](https://arxiv.org/abs/2503.10110)  [[Project Link]](https://impact-planning.github.io) [2025]
- - [ ] Collision-inclusive Manipulation Planning for Occluded Object Grasping via Compliant Robot Motions [[Paper Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11091413) [2025]
- - [ ] Harnessing the Synergy between Pushing, Grasping, and Throwing to Enhance Object Manipulation in Cluttered Scenarios [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10610548) [2024]
- - [x] GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10265134)  [[Project Link]](https://github.com/mkt1412/GraspGPT_public) [2023]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x]  [[Paper Link]]()  [[Project Link]]()  [2024] -->

# 🏆 **VLA Model**

- - [x] RDT-1B: A DIFFUSION FOUNDATION MODEL FOR BIMANUAL MANIPULATION [[Paper Link]](https://arxiv.org/pdf/2410.07864) [[Project Link]](https://github.com/thu-ml/RoboticsDiffusionTransformer) [2024]
- - [x] π0: A Vision-Language-Action Flow Model for General Robot Control [[Paper Link]](https://arxiv.org/pdf/2410.24164) [[Project Link]](https://github.com/Physical-Intelligence/openpi) [2024]
- - [x] DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes [[Paper Link]](https://arxiv.org/pdf/2410.23004) [[Project Link]](https://github.com/PKU-EPIC/DexGraspNet2) [2024]
- - [x] Yell At Your Robot: Improving On-the-Fly from Language Corrections [[Paper Link]](https://arxiv.org/abs/2403.12910) [[Project Link]](https://github.com/yay-robot/yay_robot) [2024]
- - [x] Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2209.05451) [[Project Link]](https://github.com/peract/peract#download) [2022]
- - [x] Q-attention: Enabling Efficient Learning for Vision-based Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2105.14829) [[Project Link]](https://github.com/stepjam/ARM) [2022]
- - [x] RVT: Robotic View Transformer for 3D Object Manipulation [[Paper Link]](https://arxiv.org/pdf/2306.14896) [[Project Link]](https://github.com/NVlabs/RVT/tree/0b170d7f1e27a13299a5a06134eeb9f53d494e54) [2023]
- - [ ] UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent [[Paper Link]](https://arxiv.org/pdf/2501.18867) [2025]
- - [x] Universal Actions for Enhanced Embodied Foundation Models [[Paper Link]](https://arxiv.org/abs/2501.10105) [[Project Link]](https://github.com/2toinf/UniAct) [2025]
- - [x] OpenVLA: An Open-Source Vision-Language-Action Model [[Paper Link]](https://arxiv.org/abs/2406.09246) [[Project Link]](https://openvla.github.io) [2024]
- - [x] AnyPlace: Learning Generalized Object Placement for Robot Manipulation [[Paper Link]](https://www.arxiv.org/abs/2502.04531) [[Project Link]](https://any-place.github.io/) [2025]
- - [x] Robotic Control via Embodied Chain-of-Thought Reasoning [[Paper Link]](https://arxiv.org/abs/2407.08693) [[Project Link]](https://embodied-cot.github.io) [2024]
- - [x] Language-Guided Object-Centric Diffusion Policy for Collision-Aware Robotic Manipulation [[Paper Link]](https://arxiv.org/pdf/2407.00451) [2024]
- - [x] Hierarchical Diffusion Policy: manipulation trajectory generation via contact guidance [[Paper Link]](https://arxiv.org/pdf/2411.12982) [[Project Link]](https://github.com/dexin-wang/Hierarchical-Diffusion-Policy) [2024]
- - [x] DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control [[Paper Link]](https://arxiv.org/abs/2502.05855) [[Project Link]](https://github.com/juruobenruo/DexVLA) [2025]
- - [ ] RoboGrasp: A Universal Grasping Policy for Robust Robotic Control [[Paper Link]](https://arxiv.org/abs/2502.03072) [2025]
- - [ ] Improving Vision-Language-Action Model with Online Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/2501.16664) [2025]
- - [ ] RoboHorizon: An LLM-Assisted Multi-View World Model for Long-Horizon Robotic Manipulation [[Paper Link]](https://arxiv.org/abs/2501.06605) [2025]
- - [x] Equivariant Diffusion Policy [[Paper Link]](https://arxiv.org/abs/2407.01812) [[Project Link]](https://equidiff.github.io) [2024]
- - [x] FAST: Efficient Action Tokenization for Vision-Language-Action Models [[Paper Link]](https://arxiv.org/abs/2501.09747) [[Project Link]](https://www.pi.website/research/fast) [2025]
- - [ ] Gemini Robotics: Bringing AI into the Physical World [[Paper Link]](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf) [2025]
- - [x] Robotic Control via Embodied Chain-of-Thought Reasoning [[Paper Link]](https://arxiv.org/abs/2407.08693) [[Project Link]](https://embodied-cot.github.io) [2025]
- - [ ] RT-H: Action Hierarchies Using Language [[Paper Link]](https://arxiv.org/abs/2403.01823) [[Project Link]](https://rt-hierarchy.github.io) [2024]
- - [ ] AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems [[Paper Link]](https://arxiv.org/abs/2503.06669) [[Project Link]](https://agibot-world.com) [2025]
- - [x] OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics [[Paper Link]](https://arxiv.org/abs/2401.12202)  [[Project Link]](https://ok-robot.github.io)  [2024]
- - [x] VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models[[Paper Link]](https://arxiv.org/abs/2307.05973)  [[Project Link]](https://voxposer.github.io/)  [2023]
- - [x] ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation[[Paper Link]](https://arxiv.org/abs/2409.01652)  [[Project Link]](https://rekep-robot.github.io/)  [2024]
- - [x] Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models[[Paper Link]](https://arxiv.org/abs/2410.00231)  [[Project Link]](https://helpful-doggybot.github.io/)  [2024]
- - [x] Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning[[Paper Link]](https://arxiv.org/abs/2311.17842)  [[Project Link]](https://robot-vila.github.io/)  [2023]
- - [x] Octo: An Open-Source Generalist Robot Policy[[Paper Link]](https://arxiv.org/abs/2405.12213v2)  [[Project Link]](https://octo-models.github.io)  [2024]
- - [x] Vision-Language Foundation Models as Effective Robot Imitators[[Paper Link]](https://arxiv.org/abs/2311.01378)  [[Project Link]](https://roboflamingo.github.io)  [2023]
- - [x] RT-1: Robotics Transformer for Real-World Control at Scale [[Paper Link]](https://arxiv.org/abs/2212.06817) [[Project Link]](https://robotics-transformer1.github.io/) [2022]
- - [x] PaLM-E: An Embodied Multimodal Language Model [[Paper Link]](https://arxiv.org/abs/2303.03378) [[Project Link]](https://palm-e.github.io/) [2023]
- - [x] RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control [[Paper Link]](https://arxiv.org/abs/2307.15818) [[Project Link]](https://robotics-transformer2.github.io/) [2023]
- - [x] ALOHA Unleashed: A Simple Recipe for Robot Dexterity [[Paper Link]](https://arxiv.org/abs/2410.13126) [[Project Link]](https://aloha-unleashed.github.io/) [2024]
- - [ ] Learning Universal Policies via Text-Guided Video Generation [[Paper Link]](https://arxiv.org/abs/2302.00111) [2023]
- - [x] Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation [[Paper Link]](https://arxiv.org/abs/2312.13139) [[Project Link]](https://github.com/bytedance/GR-1) [2023]
- - [x] GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation [[Paper Link]](https://arxiv.org/abs/2410.06158) [[Project Link]](https://gr2-manipulation.github.io/) [2024]
- - [x] Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers [[Paper Link]](https://arxiv.org/abs/2409.20537) [[Project Link]](https://github.com/liruiw/hpt) [2024]
- - [x] RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation [[Paper Link]](https://arxiv.org/abs/2410.07864) [[Project Link]](https://rdt-robotics.github.io/rdt-robotics/) [2024]
- - [ ] VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation [[Paper Link]](https://arxiv.org/abs/2505.09577) [2025]
- - [ ] Chain-of-Modality: Learning Manipulation Programs from Multimodal Human Videos with Vision-Language-Models [[Paper Link]](https://arxiv.org/abs/2504.13351) [2025]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 👑 **E-AI-RL**

- - [x] Aligning Diffusion Behaviors with Q-functions for Efficient Continuous Control [[Paper Link]](https://arxiv.org/abs/2407.09024) [[Project Link]](https://github.com/thu-ml/Efficient-Diffusion-Alignment) [2024]
- - [ ] MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/2410.14972) [[Project Link]](https://suninghuang19.github.io/mentor_page/) [2024]
- - [x] Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/2410.21845)  [[Project Link]](https://hil-serl.github.io/)  [2024]
- - [x] Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation [[Paper Link]](https://arxiv.org/abs/2503.02881) [[Project Link]](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) [2025]
- - [ ] Adaptive Wiping: Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10752344) [2024]

<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x]  [[Paper Link]]()  [[Project Link]]()  [2024] -->

# 🏁 **Interactive**

- - [x] Learning to Learn Faster from Human Feedback with Language Model Predictive Control [[Paper Link]](https://arxiv.org/abs/2402.11450) [[Project Link]](https://robot-teaching.github.io) [2024]
- - [ ] ELEGNT: Expressive and Functional Movement Design for Non-anthropomorphic Robot [[Paper Link]](https://arxiv.org/abs/2501.12493) [[Project Link]](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement) [2025]
- - [ ] Generative Expressive Robot Behaviors using Large Language Models [[Paper Link]](https://arxiv.org/abs/2401.14673) [[Project Link]](https://generative-expressive-motion.github.io/) [2024]
- - [ ] A Generative Model to Embed Human Expressivity into Robot Motions [[Paper Link]](https://www.mdpi.com/1424-8220/24/2/569) [2024]
- - [ ] Exploring the Design Space of Extra-Linguistic Expression for Robots [[Paper Link]](https://arxiv.org/abs/2306.15828) [2023]
- - [ ] Collection of Metaphors for Human-Robot Interaction [[Paper Link]](https://dl.acm.org/doi/10.1145/3461778.3462060) [2021]

- - [x] RHINO: Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations [[Paper Link]](https://arxiv.org/abs/2502.13134) [[Project Link]](https://humanoid-interaction.github.io/) [2025]
- - [x] ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills [[Paper Link]](https://arxiv.org/abs/2502.01143) [[Project Link]](https://agile.human2humanoid.com/) [2025]
- - [x] ExBody2: Advanced Expressive Humanoid Whole-Body Control [[Paper Link]](https://arxiv.org/abs/2412.13196) [[Project Link]](https://exbody2.github.io/) [2024]
- - [x] Expressive Whole-Body Control for Humanoid Robots [[Paper Link]](https://arxiv.org/abs/2402.16796) [[Project Link]](https://expressive-humanoid.github.io/) [2024]
- - [x] HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots [[Paper Link]](https://arxiv.org/abs/2410.21229) [[Project Link]](https://hover-versatile-humanoid.github.io/) [2024]
- - [x] OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning [[Paper Link]](https://arxiv.org/abs/2406.08858) [[Project Link]](https://omni.human2humanoid.com/) [2024]
- - [x] Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation [[Paper Link]](https://arxiv.org/abs/2403.04436) [[Project Link]](https://human2humanoid.com/) [2024]
- - [x] Learning from Massive Human Videos for Universal Humanoid Pose Control [[Paper Link]](https://arxiv.org/abs/2412.14172) [[Project Link]](https://usc-gvl.github.io/UH-1/) [2024]
- - [x] Mobile-TeleVision: Predictive Motion Priors for Humanoid Whole-Body Control [[Paper Link]](https://arxiv.org/abs/2412.07773) [[Project Link]](https://mobile-tv.github.io/) [2024]
- - [x] HumanPlus: Humanoid Shadowing and Imitation from Humans [[Paper Link]](https://arxiv.org/abs/2406.10454) [[Project Link]](https://humanoid-ai.github.io/) [2024]
- - [x] Mobile-TeleVision: Predictive Motion Priors for Humanoid Whole-Body Control [[Paper Link]](https://arxiv.org/abs/2412.07773) [[Project Link]](https://mobile-tv.github.io/) [2024]
- - [ ] Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration [[Paper Link]](https://arxiv.org/abs/2502.14795) [2025]
- - [ ] XBG: End-to-End Imitation Learning for Autonomous Behaviour in Human-Robot Interaction and Collaboration [[Paper Link]](https://arxiv.org/abs/2406.15833) [2024]
- - [ ] EMOTION: Expressive Motion Sequence Generation for Humanoid Robots with In-Context Learning [[Paper Link]](https://arxiv.org/abs/2410.23234) [[Project Link]](https://machinelearning.apple.com/research/emotion-expressive-motion) [2024]
- - [ ] HARMON: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions [[Paper Link]](https://arxiv.org/abs/2410.12773) [[Project Link]](https://ut-austin-rpl.github.io/Harmon/) [2024]
- - [ ] ImitationNet: Unsupervised Human-to-Robot Motion Retargeting via Shared Latent Space [[Paper Link]](https://arxiv.org/abs/2309.05310) [[Project Link]](https://evm7.github.io/UnsH2R/) [2023]

- - [ ] FABG : End-to-end Imitation Learning for Embodied Affective Human-Robot Interaction [[Paper Link]](https://arxiv.org/abs/2503.01363) [[Project Link]](https://cybergenies.github.io/) [2025]
- - [ ] HAPI: A Model for Learning Robot Facial Expressions from Human Preferences [[Paper Link]](https://arxiv.org/abs/2503.17046) [2025]
- - [ ] Human-robot facial coexpression [[Paper Link]](https://www.science.org/doi/epdf/10.1126/scirobotics.adi4724) [2024]
- - [ ] Unlocking Human-Like Facial Expressions in Humanoid Robots: A Novel Approach for Action Unit Driven Facial Expression Disentangled Synthesis [[Paper Link]](https://ieeexplore.ieee.org/document/10582526) [2024]
- - [x] UGotMe: An Embodied System for Affective Human-Robot Interaction [[Paper Link]](https://arxiv.org/abs/2410.18373) [[Project Link]](https://lipzh5.github.io/HumanoidVLE/) [2024]
- - [ ] Knowing Where to Look: A Planning-based Architecture to Automate the Gaze Behavior of Social Robots* [[Paper Link]](https://arxiv.org/abs/2210.02866) [2022]
- - [ ] Naturalistic Head Motion Generation from Speech [[Paper Link]](https://arxiv.org/abs/2210.14800v1) [2022]
- - [ ] Transitioning to Human Interaction with AI Systems: New Challenges and Opportunities for HCI Professionals to Enable Human-Centered AI [[Paper Link]](https://www.researchgate.net/publication/359771448_Transitioning_to_Human_Interaction_with_AI_Systems_New_Challenges_and_Opportunities_for_HCI_Professionals_to_Enable_Human-Centered_AI) [2023]
- - [ ] Roots and Requirements for Collaborative AI [[Paper Link]](https://arxiv.org/pdf/2303.12040v1) [2023]
- - [ ] From Human-Computer Interaction to Human-AI Interaction:New Challenges and Opportunities for Enabling Human-Centered AI [[Paper Link]](https://arxiv.org/vc/arxiv/papers/2105/2105.05424v1.pdf) [2021]
- - [ ] From explainable to interactive AI: A literature review on current trends in human-AI interaction [[Paper Link]](https://www.researchgate.net/publication/380836066_From_explainable_to_interactive_AI_A_literature_review_on_current_trends_in_human-AI_interaction#fullTextFileContent) [2024]
- - [ ] Treat robots as humans? Perspective choice in human-human and human-robot spatial language interaction [[Paper Link]](https://www.researchgate.net/publication/371843516_Treat_robots_as_humans_Perspective_choice_in_human-human_and_human-robot_spatial_language_interaction) [2023]
- - [ ] Advances in Large Language Models for Robotics [[Paper Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10809099) [2024]
- - [ ] Grounding Language to Natural Human-Robot Interaction in Robot Navigation Tasks [[Paper Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9739344) [2021]
- - [ ] Multi-modal interaction with transformers: bridging robots and human with natural language [[Paper Link]](https://www.researchgate.net/publication/375604123_Multi-modal_interaction_with_transformers_bridging_robots_and_human_with_natural_language) [2024]
- - [ ] Robot Control Platform for Multimodal Interactions with Humans Based on ChatGPT [[Paper Link]](https://www.researchgate.net/publication/383900674_Robot_Control_Platform_for_Multimodal_Interactions_with_Humans_Based_on_ChatGPT) [2024]
- - [ ] Multi-Grained Multimodal Interaction Network for Sentiment Analysis [[Paper Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446351) [2024]
- - [ ] Vision-Language Navigation with Embodied Intelligence: A Survey [[Paper Link]](https://i-newcar.com/uploads/ueditor/20250124/2-25012414333EV.pdf) [2024] 
- - [ ] SweepMM: A High-Quality Multimodal Dataset for Sweeping Robots in Home Scenarios for Vision-Language Model [[Paper Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10447940) [2024]
- - [ ] Recent advancements in multimodal human–robot interaction [[Paper Link]](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1084000/full) [2023]
- - [ ] Multi-Modal Data Fusion in Enhancing Human-Machine Interaction for Robotic Applications: A Survey [[Paper Link]](https://www.researchgate.net/publication/358655503_Multi-Modal_Data_Fusion_in_Enhancing_Human-Machine_Interaction_for_Robotic_Applications_A_Survey) [2022]
- - [ ] LaMI: Large Language Models for Multi-Modal Human-Robot Interaction [[Paper Link]](https://dl.acm.org/doi/pdf/10.1145/3613905.3651029) [2024]
- - [ ] "Help Me Help the AI": Understanding How Explainability Can Support Human-AI Interaction [[Paper Link]](https://www.researchgate.net/publication/364531951_Help_Me_Help_the_AI_Understanding_How_Explainability_Can_Support_Human-AI_Interaction) [2022]
- - [ ] Employing Co-Learning to Evaluate the Explainability of Multimodal Sentiment Analysis [[Paper Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9792204) [2024]
- - [ ] Towards Responsible AI: Developing Explanations to Increase Human-AI Collaboration [[Paper Link]](https://www.researchgate.net/publication/371818964_Towards_Responsible_AI_Developing_Explanations_to_Increase_Human-AI_Collaboration) [2023]
- - [ ] Toward Affective XAI: Facial Affect Analysis for Understanding Explainable Human-AI Interactions [[Paper Link]](https://www.researchgate.net/publication/352480152_Toward_Affective_XAI_Facial_Affect_Analysis_for_Understanding_Explainable_Human-AI_Interactions) [2021]

<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 💻 **Simulator**

- - [x] ORBIT: A Unified Simulation Framework for Interactive Robot Learning Environments [[Paper Link]](https://arxiv.org/abs/2301.04195) [[Project Link]](https://github.com/isaac-sim/IsaacLab) [2023]
- - [ ] Gazebo [[Paper Link]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=79f91c1c95271a075b91e9fdca43d6c31e4cbe17) [[Project Link]](https://gazebosim.org/home) [2004]
- - [ ] Pybullet, a python module for physics simulation for games, robotics and machine learning [[Project Link]](https://github.com/bulletphysics/bullet3) [2021]
- - [x] Mujoco: A physics engine for model-based control [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/6386109) [[Project Link]](https://github.com/google-deepmind/mujoco) [2012]
- - [ ] V-REP: A versatile and scalable robot simulation framework [[Project Link]](https://www.coppeliarobotics.com) [2013]
- - [x] AI2-THOR: An Interactive 3D Environment for Visual AI [[Paper Link]](https://arxiv.org/abs/1712.05474) [[Project Link]](https://ai2thor.allenai.org) [2017]
- - [x] CLIPORT: What and Where Pathways for Robotic Manipulation [[Paper Link]](https://arxiv.org/pdf/2109.12098.pdf) [[Project Link]](https://github.com/cliport/cliport?tab=readme-ov-file) [2021]
- - [x] BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation [[Paper Link]](https://arxiv.org/abs/2403.09227) [[Project Link]](https://github.com/StanfordVL/OmniGibson) [2024]
- - [x] RLBench: The Robot Learning Benchmark & Learning Environment [[Paper Link]](https://arxiv.org/abs/1909.12271) [[Project Link]](https://github.com/stepjam/RLBench) [2019]
- - [x] MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations [[Paper Link]](https://arxiv.org/abs/2310.17596) [[Project Link]](https://github.com/NVlabs/mimicgen_environments) [2023]
- - [x] CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks [[Paper Link]](https://arxiv.org/pdf/2112.03227) [[Project Link]](https://github.com/mees/calvin) [2022]
- - [x] Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning [[Paper Link]](https://arxiv.org/abs/1910.10897) [[Project Link]](https://github.com/Farama-Foundation/Metaworld) [2019]
- - [x] ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI [[Paper Link]](https://arxiv.org/abs/2410.00425) [[Project Link]](https://github.com/haosulab/ManiSkill) [2024]
- - [x] HomeRobot: Open-Vocabulary Mobile Manipulation [[Paper Link]](https://arxiv.org/abs/2306.11565) [[Project Link]](https://github.com/facebookresearch/home-robot) [2023]
- - [x] ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes [[Paper Link]](http://arxiv.org/abs/2304.04321) [[Project Link]](https://github.com/arnold-benchmark/arnold) [2023]
- - [x] Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots [[Paper Link]](https://arxiv.org/abs/2310.13724) [[Project Link]](https://github.com/facebookresearch/habitat-lab) [2023]
- - [x] InfiniteWorld: A Unified Scalable Simulation Framework for General Visual-Language Robot Interaction [[Paper Link]](https://arxiv.org/pdf/2412.05789) [[Project Link]](https://github.com/pzhren/InfiniteWorld) [2024]
- - [ ] ProcTHOR: Large-Scale Embodied AI Using Procedural Generation [[Paper Link]](https://arxiv.org/abs/2206.06994) [[Project Link]](https://procthor.allenai.org/#explore) [2022]
- - [x] Holodeck: Language Guided Generation of 3D Embodied AI Environments [[Paper Link]](https://arxiv.org/pdf/2312.09067) [[Project Link]](https://github.com/allenai/Holodeck) [2023]
- - [x] PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI [[Paper Link]](https://arxiv.org/abs/2404.09465.pdf) [[Project Link]](https://github.com/PhyScene/PhyScene/tree/main) [2024]
- - [x] RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation [[Paper Link]](https://arxiv.org/pdf/2311.01455) [[Project Link]](https://robogen-ai.github.io) [2023]
- - [x] Genesis: A Universal and Generative Physics Engine for Robotics and Beyond [[Project Link]](https://github.com/Genesis-Embodied-AI/Genesis) [2025]
- - [x] Webots: open-source robot simulator [[Paper Link]](https://cyberbotics.com/doc/reference/index) [[Project Link]](https://github.com/cyberbotics/webots) [2018]
- - [x] Unity: A General Platform for Intelligent Agents [[Paper Link]](https://arxiv.org/pdf/1809.02627) [[Project Link]](https://github.com/Unity-Technologies/ml-agents) [2020]
- - [x] ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation [[Paper Link]](https://arxiv.org/pdf/2007.04954) [[Project Link]](https://github.com/threedworld-mit/tdw) [2021]
- - [x] iGibson 1.0: A Simulation Environment for Interactive Tasks in Large Realistic Scenes [[Paper Link]](https://arxiv.org/pdf/2012.02924) [[Project Link]](https://svl.stanford.edu/igibson/) [2021]
- - [x] SAPIEN: A SimulAted Part-based Interactive ENvironment [[Paper Link]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.pdf) [[Project Link]](https://sapien.ucsd.edu) [2020]
- - [ ] VirtualHome: Simulating Household Activities via Programs [[Paper Link]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Puig_VirtualHome_Simulating_Household_CVPR_2018_paper.pdf) [[Project Link]](http://virtual-home.org) [2018]
- - [ ] Modular Open Robots Simulation Engine: MORSE [[Paper Link]](https://www.openrobots.org/morse/material/media/pdf/paper-icra.pdf) [[Project Link]](https://github.com/morse-simulator/morse) [2011]
- - [x] VRKitchen: an Interactive 3D Virtual Environment for Task-oriented Learning [[Paper Link]](https://arxiv.org/pdf/1903.05757) [[Project Link]](https://github.com/xfgao/VRKitchen) [2019]
- - [x] CHALET: Cornell House Agent Learning Environment [[Paper Link]](https://arxiv.org/pdf/1801.07357) [[Project Link]](https://github.com/lil-lab/chalet) [2018]
- - [x] Habitat: A Platform for Embodied AI Research [[Paper Link]](https://arxiv.org/pdf/1904.01201) [[Project Link]](https://github.com/facebookresearch/habitat-sim) [2019]
- - [x] MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge [[Paper Link]](https://arxiv.org/abs/2206.08853) [[Project Link]](https://github.com/MineDojo/MineDojo) [2022]
- - [x] ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks [[Paper Link]](https://arxiv.org/abs/1912.01734) [[Project Link]](https://github.com/askforalfred/alfred) [2019]
- - [x] BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning [[Paper Link]](https://arxiv.org/pdf/1810.08272) [[Project Link]](https://github.com/mila-iqia/babyai/tree/iclr19) [2019]
- - [x] Gibson Env: Real-World Perception for Embodied Agents [[Paper Link]](https://arxiv.org/pdf/1808.10654) [[Project Link]](http://gibsonenv.stanford.edu) [2018]
- - [x] iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks [[Paper Link]](https://arxiv.org/abs/2108.03272) [[Project Link]](https://svl.stanford.edu/igibson/) [2021]
- - [x] RoboTHOR: An Open Simulation-to-Real Embodied AI Platform [[Paper Link]](https://arxiv.org/pdf/2004.06799) [[Project Link]](https://ai2thor.allenai.org/robothor/) [2020]
- - [x] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning [[Paper Link]](https://arxiv.org/pdf/2306.03310) [[Project Link]](https://github.com/Lifelong-Robot-Learning/LIBERO) [2023]
- - [x] robosuite: A Modular Simulation Framework and Benchmark for Robot Learning [[Paper Link]](https://robosuite.ai/assets/whitepaper.pdf) [[Project Link]](https://github.com/ARISE-Initiative/robosuite) [2020]
- - [ ] Demonstrating HumanTHOR: A Simulation Platform and Benchmark for Human-Robot Collaboration in a Shared Workspace [[Paper Link]](https://arxiv.org/abs/2406.06498) [[Project Link]](https://sites.google.com/view/humanthor/) [2024]
- - [x] Robomimic: What Matters in Learning from Offline Human Demonstrations for Robot Manipulation [[Paper Link]](https://arxiv.org/abs/2108.03298) [[Project Link]](https://github.com/ARISE-Initiative/robomimic) [2021]
- - [x] Adroit: Manipulators and Manipulation in high dimensional spaces [[Paper Link]](https://digital.lib.washington.edu/researchworks/items/f810e199-f3fe-4918-8603-65790e0fdc16) [[Project Link]](https://github.com/vikashplus/Adroit) [2016]
- - [x] Gymnasium-Robotics [[Paper Link]](https://robotics.farama.org/release_notes/) [[Project Link]](https://github.com/Farama-Foundation/Gymnasium-Robotics) [2024]
- - [x] RoboHive: A Unified Framework for Robot Learning [[Paper Link]](https://arxiv.org/abs/2310.06828) [[Project Link]](https://sites.google.com/view/robohive) [2024]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 📊 **Dataset**

- - [ ] Efficient Grasping from RGBD Images: Learning using a new Rectangle Representation [[Paper Link]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3c104b0e182a5f514d3aebecc93629bbcf1434ac) [2011]
- - [x] Real-World Multiobject, Multigrasp Detection [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/8403246) [[Project Link]](https://github.com/ivalab/grasp_multiObject_multiGrasp) [2018]
- - [x] Jacquard: A Large Scale Dataset for Robotic Grasp Detection [[Paper Link]](https://arxiv.org/pdf/1803.11469) [[Project Link]](https://jacquard.liris.cnrs.fr/) [2018]
- - [x] Learning 6-DOF Grasping Interaction via Deep Geometry-aware 3D Representations [[Paper Link]](https://arxiv.org/pdf/1708.07303) [[Project Link]](https://research.google/pubs/learning-6-dof-grasping-interaction-via-deep-3d-geometry-aware-representations/) [2018]
- - [x] ACRONYM: A Large-Scale Grasp Dataset Based on Simulation [[Paper Link]](https://arxiv.org/pdf/2011.09584) [[Project Link]](https://sites.google.com/nvidia.com/graspdataset) [2020]
- - [x] EGAD! an Evolved Grasping Analysis Dataset for diversity and reproducibility in robotic manipulation [[Paper Link]](https://arxiv.org/pdf/2003.01314) [[Project Link]](https://dougsm.github.io/egad/) [2020]
- - [x] GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping [[Paper Link]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf) [[Project Link]](https://github.com/RayYoh/OCRM_survey/blob/main/www.graspnet.net) [2020]
- - [x] Grasp-Anything: Large-scale Grasp Dataset from Foundation Models [[Paper Link]](https://arxiv.org/pdf/2309.09818) [[Project Link]](https://airvlab.github.io/grasp-anything/) [2023]
- - [x] DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes [[Paper Link]](https://openreview.net/pdf?id=5W0iZR9J7h) [[Project Link]](https://github.com/PKU-EPIC/DexGraspNet2) [2024]
- - [x] Yale-CMU-Berkeley dataset for robotic manipulation research [[Paper Link]](https://journals.sagepub.com/doi/pdf/10.1177/0278364917700714) [[Project Link]](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) [2017]
- - [x] AKB-48: A Real-World Articulated Object Knowledge Base [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_AKB-48_A_Real-World_Articulated_Object_Knowledge_Base_CVPR_2022_paper.pdf) [[Project Link]](https://liuliu66.github.io/AKB-48/) [2022]
- - [x] GAPartNet: Cross-Category Domain-Generalizable Object Perceptionmand Manipulation via Generalizable and Actionable Parts [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2023/papers/Geng_GAPartNet_Cross-Category_Domain-Generalizable_Object_Perception_and_Manipulation_via_Generalizable_and_CVPR_2023_paper.pdf) [[Project Link]](https://pku-epic.github.io/GAPartNet) [2022]
- - [x] Bi-DexHands: Towards Human-Level Bimanual Dexterous Manipulation [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10343126) [[Project Link]](https://github.com/PKU-MARL/DexterousHands) [2022]
- - [x] DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2023/papers/Bao_DexArt_Benchmarking_Generalizable_Dexterous_Manipulation_With_Articulated_Objects_CVPR_2023_paper.pdf) [[Project Link]](https://www.chenbao.tech/dexart/) [2023]
- - [x] PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2023/papers/Geng_PartManip_Learning_Cross-Category_Generalizable_Part_Manipulation_Policy_From_Point_Cloud_CVPR_2023_paper.pdf) [[Project Link]](https://pku-epic.github.io/PartManip) [2023]
- - [x] Open X-Embodiment: Robotic Learning Datasets and RT-X Models [[Paper Link]](https://arxiv.org/abs/2310.08864) [[Project Link]](https://github.com/google-deepmind/open_x_embodiment) [2024]
- - [x] RH20T-P: A Primitive-Level Robotic Dataset Towards Composable Generalization Agents [[Paper Link]](https://arxiv.org/pdf/2403.19622) [[Project Link]](https://sites.google.com/view/rh20t-primitive/main) [2025]
- - [x] ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation [[Paper Link]](https://aloha-2.github.io/assets/aloha2.pdf) [[Project Link]](https://aloha-2.github.io) [2024]
- - [x] GRUtopia: Dream General Robots in a City at Scale [[Paper Link]](https://arxiv.org/pdf/2407.10943) [[Project Link]](https://github.com/OpenRobotLab/GRUtopia) [2024]
- - [x] All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents [[Paper Link]](https://arxiv.org/pdf/2408.10899) [[Project Link]](https://imaei.github.io/project_pages/ario/) [2024]
- - [x] VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks [[Paper Link]](https://arxiv.org/abs/2412.18194) [[Project Link]](https://vlabench.github.io) [2024]
- - [x] RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation [[Paper Link]](https://arxiv.org/abs/2412.13877) [[Project Link]](https://x-humanoid-robomind.github.io) [2024]
- - [x] On Bringing Robots Home [[Paper Link]](https://dobb-e.com/#paper) [[Project Link]](https://github.com/notmahi/dobb-e) [2023]
- - [x] Empowering Embodied Manipulation: A Bimanual-Mobile Robot Manipulation Dataset for Household Tasks [[Paper Link]](https://arxiv.org/pdf/2405.18860) [[Project Link]](https://github.com/Louis-ZhangLe/BRMData) [2024]
- - [x] DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset [[Paper Link]](https://arxiv.org/abs/2403.12945) [[Project Link]](https://droid-dataset.github.io) [2024]
- - [x] BridgeData V2: A Dataset for Robot Learning at Scale [[Paper Link]](https://arxiv.org/abs/2308.12952) [[Project Link]](https://rail-berkeley.github.io/bridgedata/) [2024]
- - [x] RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking [[Paper Link]](https://arxiv.org/pdf/2309.01918.pdf) [[Project Link]](https://robopen.github.io) [2023]
- - [x] AgiBot World Colosseum [[Paper Link]](https://github.com/OpenDriveLab/AgiBot-World) [[Project Link]](https://agibot-world.com) [2024]
- - [x] REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction [[Paper Link]](https://arxiv.org/abs/2306.15724) [[Project Link]](https://github.com/real-stanford/reflect) [2023]
- - [x] OakInk2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion [[Paper Link]](https://arxiv.org/abs/2403.19417) [[Project Link]](https://oakink.net/v2/) [2024]
- - [x] A dataset of relighted 3d interacting hands [[Paper Link]](https://arxiv.org/abs/2310.17768) [[Project Link]](https://mks0601.github.io/ReInterHand/) [2023]
- - [x] Human-agent joint learning for efficient robot manipulation skill acquisition [[Paper Link]](https://arxiv.org/abs/2407.00299) [[Project Link]](https://norweig1an.github.io/HAJL.github.io/) [2025]
- - [x] RoboNet: Large-Scale Multi-Robot Learning [[Paper Link]](https://arxiv.org/abs/1910.11215) [[Project Link]](https://www.robonet.wiki) [2020]
- - [x] MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale [[Paper Link]](https://arxiv.org/abs/2104.08212) [[Project Link]](https://karolhausman.github.io/mt-opt/) [2021]
- - [x] BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning [[Paper Link]](https://arxiv.org/abs/2202.02005) [[Project Link]](https://sites.google.com/view/bc-z/home?pli=1) [2022]
- - [x] VIMA: General Robot Manipulation with Multimodal Prompts [[Paper Link]](https://arxiv.org/abs/2210.03094) [[Project Link]](https://vimalabs.github.io) [2023]
- - [x] FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset [[Paper Link]](https://arxiv.org/abs/2409.19499) [[Project Link]](https://fastumi.com/) [2024]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 🔧 **Toolkit**

- - [x] PyRep: Bringing V-REP to Deep Robot Learning [[Paper Link]](https://arxiv.org/abs/1906.11176) [[Project Link]](https://github.com/stepjam/PyRep) [2024]
- - [x] Yet Another Robotics and Reinforcement learning framework for PyTorch [[Project Link]](https://github.com/stepjam/YARR) [2024]
<!-- - - [x]  [[Paper Link]]()  [[Project Link]]()  [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024]
- - [x] [[Paper Link]]() [[Project Link]]() [2024] -->

# 📖 **Citation**

> If you think this repository is helpful, please feel free to leave a star ⭐️

# 😊 **Acknowledgements**

Thanks for the repository:

1. [Embodied_AI_Paper_List](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List?tab=readme-ov-file#paper-list-for-embodied-ai>)
2. [OCRM_survey](https://github.com/RayYoh/OCRM_survey?tab=readme-ov-file#sim-to-real-generalization)
3. [Awesome-Generalist-Robots-via-Foundation-Models](https://github.com/JeffreyYH/Awesome-Generalist-Robots-via-Foundation-Models)
4. [Awesome-Embodied-Agent-with-LLMs](https://github.com/zchoi/Awesome-Embodied-Agent-with-LLMs?tab=readme-ov-file#survey)
5. [Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models?tab=readme-ov-file#survey)
6. [Awesome-Humanoid-Robot-Learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning)
