
def decode_vote(self, top_logits):
    ''''protocol:
    The vote is a 9-channel H*W tensor, representing the input map (larger)'s vote for being
    part of which pixel in the output map (smaller) for a BottleNeck.
    the 9-channel has different meaning for different stride.
    For stride = 1:
        channel 0 : vote value for the relative [-1, -1] pixel in the output map.
        channel 1 : vote value for the relative [-1,  0] pixel in the output map.
        channel 2 : vote value for the relative [-1,  1] pixel in the output map.
        ...
        channel 8 : vote value for the relative [ 1,  1] pixel in the output map.

    For stride = 1: pixels in the input map are divided into 4-pixel tuple
        channel 0 : pixel (0,0) vote value for the big pixel (the 4-pixel tuple) relative [0, 0] pixel in the output map.
        /- channel 1 : pixel (0,1) vote value for the big pixel (the 4-pixel tuple) relative [0, 0] pixel in the output map.
        \- channel 2 : pixel (0,1) vote value for the big pixel (the 4-pixel tuple) relative [0, 1] pixel in the output map.
        /- channel 3 : pixel (1,0) vote value for the big pixel (the 4-pixel tuple) relative [0, 0] pixel in the output map.
        \- channel 4 : pixel (1,0) vote value for the big pixel (the 4-pixel tuple) relative [1, 0] pixel in the output map.
        /- channel 5 : pixel (1,1) vote value for the big pixel (the 4-pixel tuple) relative [0, 0] pixel in the output map.
    |-- channel 6 : pixel (1,1) vote value for the big pixel (the 4-pixel tuple) relative [1, 0] pixel in the output map.
    |-- channel 7 : pixel (1,1) vote value for the big pixel (the 4-pixel tuple) relative [1, 0] pixel in the output map.
        \- channel 8 : pixel (1,1) vote value for the big pixel (the 4-pixel tuple) relative [1, 0] pixel in the output map.
    '''

    for btn_i, botnek, vote, insize in zip(range(len(self.bottlenecks))[::-1], 
                self.bottlenecks[::-1], self.votes[::-1], self.botnek_insize[::-1]):
        # N, C, H, W = insize
        N, C, H, W = top_logits.shape
        # print(vote.shape, top_logits.shape) # 9*N_v op C*N_t --> C * N_v
        padded_top_logits = F.pad(input=top_logits, pad=(1, 1, 1, 1), mode='constant', value=0)
        back_logits = top_logits.new_zeros((insize[0], 19, insize[-2], insize[-1]))

        if False:
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,0,0), save_name='bottleneck'+str(btn_i))
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,1,0), save_name='bottleneck'+str(btn_i))
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,0,1), save_name='bottleneck'+str(btn_i))
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,1,1), save_name='bottleneck'+str(btn_i))
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,0,2), save_name='bottleneck'+str(btn_i))
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,2,0), save_name='bottleneck'+str(btn_i))
            self.verify_range(self.bottlenecks[btn_i], impulse=(0,0,2,2), save_name='bottleneck'+str(btn_i))

        if botnek.stride == 1:
            for vote_i in range(9):
                dh, dw = vote_i//3, vote_i%3
                back_logits += vote[:, [vote_i]] * padded_top_logits[:,:,dh:H+dh, dw:W+dw]
        elif botnek.stride == 2:
            back_logits[:,:,0::2, 0::2] = vote[:, [0]] * padded_top_logits[:,:,1:H+1, 1:W+1]
            back_logits[:,:,0::2, 1::2] = vote[:, [1]] * padded_top_logits[:,:,1:H+1, 1:W+1] +\
                                            vote[:, [2]] * padded_top_logits[:,:,1:H+1, 2:W+2]
            back_logits[:,:,1::2, 0::2] = vote[:, [3]] * padded_top_logits[:,:,1:H+1, 1:W+1] +\
                                            vote[:, [4]] * padded_top_logits[:,:,2:H+2, 1:W+1]
            back_logits[:,:,1::2, 1::2] = vote[:, [5]] * padded_top_logits[:,:,1:H+1, 1:W+1] +\
                                            vote[:, [6]] * padded_top_logits[:,:,1:H+1, 2:W+2] +\
                                            vote[:, [7]] * padded_top_logits[:,:,2:H+2, 1:W+1] +\
                                            vote[:, [8]] * padded_top_logits[:,:,2:H+2, 2:W+2]
        else:
            import pdb; pdb.set_trace()
        
        top_logits = back_logits

    return top_logits
