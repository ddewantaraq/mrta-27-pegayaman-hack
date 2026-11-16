import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'optional neural research dependencies absent')
class TemporalResearchTests(unittest.TestCase):
    def test_numpy_dual_heads_and_adaptive_decoder_match_research(self):
        from boundary_ownership_model import BoundaryOwnershipNet,decode_boundary
        from numpy_boundary_ownership import forward_dual,decode_boundary as numpy_decode
        torch.set_num_threads(1); torch.manual_seed(37)
        net=BoundaryOwnershipNet(width=8,layers=2).eval()
        torch.nn.init.normal_(net.boundary.weight,std=.1)
        rng=np.random.RandomState(37); n=109
        x=rng.randn(n,57).astype(np.float32); nodes=rng.rand(n,2,11,8).astype(np.float32)
        nodes[...,6]=nodes[...,6]>.25; context=rng.rand(n,8).astype(np.float32)
        base=rng.choice([-1.,0.,1.],n).astype(np.float32)
        p={key:value.numpy() for key,value in net.state_dict().items()}
        p['temporal.project.weight']=p['temporal.project.weight'][:,np.r_[np.arange(57),157,np.arange(57,157)]]
        p['layers']=np.array(2)
        with torch.inference_mode():
            expected=net(torch.from_numpy(x.T.copy())[None],torch.from_numpy(nodes)[None],
                         torch.from_numpy(context)[None],torch.from_numpy(base)[None])
        team,edge=forward_dual(np.column_stack([x,base]),nodes,context,p)
        np.testing.assert_allclose(team+net.prior*base,expected[0][0].numpy(),rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(edge,expected[1][0].numpy(),rtol=2e-5,atol=2e-6)
        live=np.ones(n,bool); live[50:60]=False
        np.testing.assert_array_equal(numpy_decode(team,edge,live),decode_boundary(team,edge,live))

    def test_fine_decoder_matches_approved_at_original_sampling_rate(self):
        from temporal_decode import decode_scaled
        from dead_ball_model import decode_dead_logits
        signal=np.random.RandomState(31).randn(701)
        live=np.ones(len(signal),bool); live[170:190]=False
        period=np.r_[np.ones(301),np.full(400,2)]
        np.testing.assert_array_equal(decode_scaled(signal,live,period,stride=5),
                                      decode_dead_logits(signal,live,period))

    def test_player_geometry_aligns_at_shared_sampling_instants(self):
        from player_nodes import pack_player_nodes
        xy=np.zeros((100,2,2),np.float32)
        xy[:,0,0]=10+np.arange(100)/25.; xy[:,1,0]=90-np.arange(100)/25.
        xy[:,:,1]=34; period=np.ones(100,np.int8)
        coarse=pack_player_nodes(xy,[0,1],period,stride=5)
        fine=pack_player_nodes(xy,[0,1],period,stride=2)
        np.testing.assert_array_equal(coarse[::2],fine[::5])

    def test_fine_resolution_correction_starts_at_frozen_teacher(self):
        from highres_ownership_model import HighResolutionOwnershipNet
        torch.set_num_threads(1); torch.manual_seed(19)
        model=HighResolutionOwnershipNet(width=8,layers=3).eval()
        x=torch.randn(1,57,41); nodes=torch.rand(1,41,2,11,8); nodes[...,6]=1
        context=torch.rand(1,41,8); teacher=torch.randn(1,41)
        with torch.inference_mode():
            torch.testing.assert_close(model(x,nodes,context,teacher),teacher,atol=0,rtol=0)

    def test_boundary_decoder_reproduces_constant_transition_decoder(self):
        from boundary_ownership_model import decode_boundary
        from dead_ball_model import decode_dead_logits
        rng=np.random.RandomState(41); signal=rng.randn(1100)
        signal[300:600]+=2.; signal[700:]-=2.
        graded=np.ones(len(signal),bool); graded[500:520]=False
        period=np.r_[np.ones(650),np.full(450,2)]
        boundary=np.full(len(signal),np.log(.01/.99)+np.log(10.))
        expected=decode_dead_logits(signal,graded,period)
        np.testing.assert_array_equal(decode_boundary(signal,boundary,graded,period),expected)

    def test_boundary_targets_ignore_period_switches_and_dead_intervals(self):
        from boundary_ownership_experiment import boundary_target
        gt=np.r_[np.full(40,'H'),np.full(40,'A'),np.full(40,'D'),np.full(40,'H')]
        g=np.ones(len(gt),bool); period=np.r_[np.ones(40),np.full(120,2)]
        self.assertFalse(boundary_target(gt,g,period).any())
        period[:]=1
        target=boundary_target(gt,g,period)
        self.assertEqual(target.argmax(),40)
        self.assertEqual(target[40],1.)
        self.assertEqual(target[80],0.)

    def test_dead_correction_starts_exactly_at_teacher(self):
        from temporal_dead_model import TemporalDeadNet
        torch.set_num_threads(1); torch.manual_seed(7)
        net=TemporalDeadNet(width=8,layers=2).eval()
        x=torch.randn(1,173,31); nodes=torch.rand(1,31,2,11,8)
        nodes[...,6]=1; ctx=torch.rand(1,31,8); baseline=torch.randn(1,31)
        with torch.inference_mode():
            torch.testing.assert_close(net(x,nodes,ctx,baseline),baseline,atol=0,rtol=0)

    def test_atomic_checkpoint_restores_optimizer_and_both_rngs(self):
        from research_checkpoint import save_checkpoint,resume_checkpoint
        torch.manual_seed(17); rng=np.random.RandomState(17)
        net=torch.nn.Linear(3,1); opt=torch.optim.AdamW(net.parameters(),lr=.001)
        net(torch.ones(2,3)).sum().backward(); opt.step()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'epoch.pt'
            save_checkpoint(path,net,opt,rng,3,dict(architecture='test'))
            expected_numpy=rng.rand(5); expected_torch=torch.rand(5)
            restored=torch.nn.Linear(3,1); optimizer=torch.optim.AdamW(restored.parameters(),lr=.1)
            self.assertEqual(resume_checkpoint(path,restored,optimizer,rng),3)
            np.testing.assert_array_equal(rng.rand(5),expected_numpy)
            torch.testing.assert_close(torch.rand(5),expected_torch,atol=0,rtol=0)
            self.assertEqual(optimizer.param_groups[0]['lr'],.001)
            for key,value in net.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key],value,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
