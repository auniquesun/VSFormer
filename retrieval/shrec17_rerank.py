# run command, e.g., python shrec17_rerank.py Alex AlexRatio_sub AlexSR  
#   读代码过程中对代码中的变量进行举例，比较容易读下去，遇到的阻碍会少很多，整个读一遍就能理解这段代码在干什么了

import sys, os

# e.g., vsformer
method_name = sys.argv[1]
# e.g., test_normal
cls_ret_name = sys.argv[2]
cls_ret_dir = os.path.join('evaluator', method_name, cls_ret_name)

# e.g., test_normal_sub
subcls_ret_name = f'{cls_ret_name}_sub'
subcls_ret_dir = os.path.join('evaluator', method_name, subcls_ret_name)

final_ret_dir = os.path.join('evaluator', f'final_{method_name}', cls_ret_name)
if not os.path.exists(final_ret_dir):
    os.mkdir(final_ret_dir)

for shape in os.listdir(cls_ret_dir) :   # evaluator/vsformer/test_normal

        f = open(os.path.join(cls_ret_dir, shape))    # e.g., evaluator/vsformer/test_normal/000009
        lines1 = f.readlines()
        f.close()
        shapes = [v.split(' ')[0] for v in lines1] # shapes with same predicted category

        f = open(os.path.join(subcls_ret_dir, shape))    # e.g., evaluator/vsformer/test_normal_sub/000009
        lines2 = f.readlines()  
        f.close()
        shapes_sub = [v.split(' ')[0] for v in lines2]    # shapes with same predicted subcategory

        # open output file
        f = open(os.path.join(final_ret_dir,shape),'w')    # evaluator/final_vsformer/test_normal/000009

        # output samples included in shapes_sub
        is_included = [(v in shapes_sub) for v in shapes]
        inds = [i for i, x in enumerate(is_included) if x == True]
        for i in inds:
            f.write(lines1[i])

        # output remaining samples
        inds = [i for i, x in enumerate(is_included) if x == False]
        for i in inds:
            f.write(lines1[i])

        f.close()

print(f'--- rerank {cls_ret_name} Done! ---')