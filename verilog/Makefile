

clean:
	rm -rf xsim.dir
	rm -rf webtalk*
	rm -rf xsim_*.*
	rm -rf test.vcd
	rm -rf *.log
	rm -rf *.pb
	rm -rf *.jou
	rm -rf *.wdb
	rm -rf .Xil

gen_reg:
	perl -i regmaker.prl reg_define.txt -v reg_define.v -vh reg_define.vh -w reg_define.html

sim:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj pe.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.testbench xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl
	#vcd2fst.exe -v test.vcd -f test.fst

sim_mac:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_mac.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_mac xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl

sim_MX_cell:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_MX_cell.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_MX_cell xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl

sim_systolic_array:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_systolic_array.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_systolic_array xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_max_power:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_max_power.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_max_power xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_shifter:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_shifter.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_shifter xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_shifter_MX_cell:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_shifter_MX_cell.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_shifter_MX_cell xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_acc_shifter:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_acc_shifter.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_acc_shifter xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_acc_shifter_MX_cell:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_acc_shifter_MX_cell.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_acc_shifter_MX_cell xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl

sim_acc_deshifter:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_acc_deshifter.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_acc_deshifter xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl

sim_acc_deshifter_MX_cell:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_acc_deshifter_MX_cell.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_acc_deshifter_MX_cell xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl

sim_wgt_shifter:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_wgt_shifter.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_wgt_shifter xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_top:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_systolic_dla_top.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_systolic_dla_top xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


sim_top_8x8_wrapper:
	rm -rf xsim.dir
	rm -rf webtalk_*
	rm -rf xsim_*.*
	#xvlog --sv -f pe.f 
	#egrep.exe "\\.sv|\\.v" pe.prj > rtl.f
	#vermin -f rtl.f
	xvlog --relax -prj tb_systolic_dla_top_8x8_wrapper.prj
	#xelab testbench -s testbench
	xelab -wto cc6c4888dbca457bbb65a113dedadb70 \
	--incr --debug typical \
	--relax --mt 2 \
	-L blk_mem_gen_v8_4_1 \
	-L xil_defaultlib \
	-L unisims_ver \
	-L unimacro_ver \
	-L secureip \
	-L xpm \
	--snapshot testbench_behav xil_defaultlib.tb_systolic_dla_top_8x8_wrapper xil_defaultlib.glbl -log elaborate.log
	xsim testbench_behav -t run.tcl


gtkwave:
	gtkwave.exe -f test.vcd
