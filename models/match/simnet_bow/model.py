import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase

class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
    
    def _init_hyper_parameters(self):
        self.dict_dim = envs.get_global_env("hyper_parameters.dict_dim") 
        self.emb_dim = envs.get_global_env("hyper_parameters.emb_dim")
        self.learning_rate = envs.get_global_env("hyper_parameters.learning_rate")        
        self.hid_dim = envs.get_global_env("hyper_parameters.hid_dim")
        self.margin = envs.get_global_env("hyper_parameters.margin")

    def input_data(self, is_infer=False, **kwargs):
        q = fluid.layers.data(name = "query", 
            shape = [1], dtype = "int64", lod_level = 1)
        pt = fluid.layers.data(name = "pos_cand",
            shape = [1], dtype = "int64", lod_level = 1) 
        nt = fluid.layers.data(name = "neg_cand",
            shape = [1], dtype = "int64", lod_level = 1)
        return [q, pt, nt]

    def net(self, inputs, is_infer=False):
        q = inputs[0]
        pt = inputs[1]
        nt = inputs[2]
        self.emb_lr = self.learning_rate * 3
        is_distributed = False
        is_sparse = True
        q_emb = fluid.layers.embedding(input=q,
                                       is_distributed=is_distributed,
                                       size=[self.dict_dim, self.emb_dim],
                                       param_attr=fluid.ParamAttr(name="__emb__",
                                                                  learning_rate=self.emb_lr),
                                       is_sparse=is_sparse
                                       )
        # vsum
        q_sum = fluid.layers.sequence_pool(
            input=q_emb,
            pool_type='sum')
        q_ss = fluid.layers.softsign(q_sum)
        # fc layer after conv
        q_fc = fluid.layers.fc(input=q_ss,
                               size=self.hid_dim,
                               param_attr=fluid.ParamAttr(name="__q_fc__",
                                                          learning_rate=self.learning_rate,
                                                          initializer=fluid.initializer.Xavier()))
        # embedding
        pt_emb = fluid.layers.embedding(input=pt,
                                        is_distributed=is_distributed,
                                        size=[self.dict_dim, self.emb_dim],
                                        param_attr=fluid.ParamAttr(
                                            name="__emb__", learning_rate=self.emb_lr,
                                            initializer=fluid.initializer.Xavier()),
                                        is_sparse=is_sparse)
        # vsum
        pt_sum = fluid.layers.sequence_pool(
            input=pt_emb,
            pool_type='sum')
        pt_ss = fluid.layers.softsign(pt_sum)
        # fc layer
        pt_fc = fluid.layers.fc(input=pt_ss,
                                size=self.hid_dim,
                                param_attr=fluid.ParamAttr(
                                    name="__fc__", learning_rate=self.learning_rate, initializer=fluid.initializer.Xavier()),
                                bias_attr=fluid.ParamAttr(name="__fc_b__", initializer=fluid.initializer.Xavier()))

        # embedding
        nt_emb = fluid.layers.embedding(input=nt,
                                        is_distributed=is_distributed,
                                        size=[self.dict_dim, self.emb_dim],
                                        param_attr=fluid.ParamAttr(name="__emb__",
                                                                   learning_rate=self.emb_lr,
                                                                   initializer=fluid.initializer.Xavier()),
                                        is_sparse=is_sparse)

        # vsum
        nt_sum = fluid.layers.sequence_pool(
            input=nt_emb,
            pool_type='sum')
        nt_ss = fluid.layers.softsign(nt_sum)
        # fc layer
        nt_fc = fluid.layers.fc(input=nt_ss,
                                size=self.hid_dim,
                                param_attr=fluid.ParamAttr(
                                    name="__fc__", learning_rate=self.learning_rate, initializer=fluid.initializer.Xavier()),
                                bias_attr=fluid.ParamAttr(name="__fc_b__", initializer=fluid.initializer.Xavier()))
        cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
        cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
        # loss
        avg_cost = self.get_loss(cos_q_pt, cos_q_nt)
        
        self._cost = avg_cost
        self._metrics["LOSS"] = avg_cost

    def get_loss(self, cos_q_pt, cos_q_nt):
        loss_op1 = fluid.layers.elementwise_sub(
            fluid.layers.fill_constant_batch_size_like(input=cos_q_pt, shape=[-1, 1], value=self.margin,
                                                       dtype='float32'), cos_q_pt)
        loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
        loss_op3 = fluid.layers.elementwise_max(
            fluid.layers.fill_constant_batch_size_like(input=loss_op2, shape=[-1, 1], value=0.0,
                                                       dtype='float32'), loss_op2)
        avg_cost = fluid.layers.mean(loss_op3)
        return avg_cost

    #def train_net(self, inputs):
        
    #def infer_net(self, inputs):
    
