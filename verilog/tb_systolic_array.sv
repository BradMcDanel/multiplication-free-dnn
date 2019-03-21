module tb_systolic_array;

	logic       clk                                ;
	logic       reset                              ;

	parameter SUBARRAY_WIDTH      = 16                        ; // width of the subarray
	parameter SUBARRAY_HEIGHT     = 16                        ; // height of the subarray
	parameter NUM_DATAFLOW_PER_MX = 2                         ;
	parameter W_DATA_MUX          = clog2(NUM_DATAFLOW_PER_MX);

	function [31:0] clog2 (input [31:0] x);
		reg [31:0] x_tmp;
		begin
			x_tmp = x-1;
			for(clog2=0; x_tmp>0; clog2=clog2+1) begin
				x_tmp = x_tmp >> 1;
			end
		end
	endfunction

	logic [SUBARRAY_WIDTH*SUBARRAY_HEIGHT*W_DATA_MUX-1:0] dataflow_select  = {(SUBARRAY_WIDTH*SUBARRAY_HEIGHT*W_DATA_MUX){1'b0}};
	logic [                        4*SUBARRAY_HEIGHT-1:0] accumulation     = {(4*SUBARRAY_HEIGHT){1'b0}}                        ;
	logic [       NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH-1:0] dataflow_in      = {(NUM_DATAFLOW_PER_MX*SUBARRAY_WIDTH){1'b0}}       ;
	logic [           SUBARRAY_WIDTH*SUBARRAY_HEIGHT-1:0] control1         = {(SUBARRAY_WIDTH*SUBARRAY_HEIGHT){1'b0}}           ;
	logic [                           SUBARRAY_WIDTH-1:0] update_w         = {(SUBARRAY_WIDTH){1'b0}}                           ;
	logic [                        4*SUBARRAY_HEIGHT-1:0] mac_en           = {(4*SUBARRAY_HEIGHT){1'b0}}                        ;
	logic [                        4*SUBARRAY_HEIGHT-1:0] clr_and_plus_one = {(4*SUBARRAY_HEIGHT){1'b0}}                        ;
	wire  [                        4*SUBARRAY_HEIGHT-1:0] result                                                                ;
	wire  [                        4*SUBARRAY_HEIGHT-1:0] result_en                                                             ;

	int        cur_weight[SUBARRAY_WIDTH-1:0][SUBARRAY_HEIGHT-1:0];
	int        cur_acc   [SUBARRAY_HEIGHT-1:0][3:0];
	//reg [7:0]  cur_dat   [SUBARRAY_WIDTH-1:0] [3:0];
	reg [7:0]  cur_dat   [SUBARRAY_WIDTH-1:0] [3:0] [4*SUBARRAY_HEIGHT-1:0];


	logic [4*SUBARRAY_HEIGHT-1:0] accumulation_in;

j_systolic_array #(
	.SUBARRAY_WIDTH     (SUBARRAY_WIDTH     ),
	.SUBARRAY_HEIGHT    (SUBARRAY_HEIGHT    ),
	.NUM_DATAFLOW_PER_MX(NUM_DATAFLOW_PER_MX)
) i_j_systolic_array (
	.clk             (clk             ),
	.reset           (reset           ),
	.accumulation_in (accumulation    ),
	.dataflow_in     (dataflow_in     ),
	.dataflow_select (dataflow_select ),
	.clr_and_plus_one(clr_and_plus_one),
	.mac_en          (mac_en          ),
	.update_w        (update_w        ),
	.control1        (control1        ),
	.result          (result          ),
	.result_en       (result_en       )
);


	initial begin
	  $dumpfile("test.vcd");
	  $dumpvars;
	end

	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

	initial begin
		reset = 1'b0;
		@(posedge clk);
		reset = 1'b1;
		@(posedge clk);
		reset = 1'b0;
	end

	// serial part
	task automatic get_clk();
		@(posedge clk);
	endtask : get_clk


	task automatic put_data_i(input data, input is_update_w, input int idx_w, input int col);
		update_w   [idx_w                        ] = is_update_w;
		dataflow_in[idx_w*NUM_DATAFLOW_PER_MX+col] = data;
		@(posedge clk);
	endtask : put_data_i

	task automatic put_acc_i(input acc_in, input int idx_h, input int idx);
		accumulation[idx_h*4 + idx]    = acc_in;
		@(posedge clk);
	endtask : put_acc_i	

	task automatic put_clr_and_plus_one(input one, input int idx_h, input int idx);
		clr_and_plus_one[4*idx_h + idx]    = one;
		@(posedge clk);
	endtask : put_clr_and_plus_one

	task automatic put_control1(input control1_i, input int idx_w, input int idx_h);
		control1[idx_h*SUBARRAY_WIDTH+idx_w]    = control1_i;
	endtask : put_control1

	task automatic get_result_bit(output logic result_o, input int idx_h, input int idx);
		@(posedge clk);
		result_o = result[4*idx_h+idx];
		//$display($stime, "timer = %d, result_o = %b", time_cnt, result_o);
	endtask : get_result_bit

	task automatic put_mac_en(input mac_en_i, input int idx_h, input int idx);
		mac_en[4*idx_h + idx] = mac_en_i;
		@(posedge clk);
	endtask : put_mac_en

	wire [4*SUBARRAY_HEIGHT-1:0] mac_en_dly_vec = result_en;


	// parallel part
	task automatic shift_data_in(input logic [7:0] data, input int idx_w, input int col);
		//get_clk;
		for (int i = 0; i < 8; i++) begin
			put_data_i(.data(data[i]), .is_update_w(1'b0), .idx_w(idx_w), .col(col));
		end
	endtask : shift_data_in

	task automatic shift_weight_in(input int weight, input int idx_w);
		reg [7:0] weight_abs;
		int       div_step;
		//cur_weight[idx_w][idx_h] = weight;
		weight_abs = (weight < 0) ? -weight : weight;
		//put_control1(.control1_i(~weight[7]), .idx_w(idx_w), .idx_h(idx_h));
		div_step = (8/NUM_DATAFLOW_PER_MX);

		if(div_step == 1) begin
			fork
				put_data_i(.data(weight_abs[0]), .is_update_w(1'b1), .idx_w(idx_w), .col(0));
				put_data_i(.data(weight_abs[1]), .is_update_w(1'b1), .idx_w(idx_w), .col(1));
				put_data_i(.data(weight_abs[2]), .is_update_w(1'b1), .idx_w(idx_w), .col(2));
				put_data_i(.data(weight_abs[3]), .is_update_w(1'b1), .idx_w(idx_w), .col(3));
				put_data_i(.data(weight_abs[4]), .is_update_w(1'b1), .idx_w(idx_w), .col(4));
				put_data_i(.data(weight_abs[5]), .is_update_w(1'b1), .idx_w(idx_w), .col(5));
				put_data_i(.data(weight_abs[6]), .is_update_w(1'b1), .idx_w(idx_w), .col(6));
				put_data_i(.data(weight_abs[7]), .is_update_w(1'b1), .idx_w(idx_w), .col(7));
			join
		end

		if(div_step == 2) begin
			for (int i = 0; i < 2; i++) begin
				fork
					put_data_i(.data(weight_abs[4*i+0]), .is_update_w(1'b1), .idx_w(idx_w), .col(0));
					put_data_i(.data(weight_abs[4*i+1]), .is_update_w(1'b1), .idx_w(idx_w), .col(1));
					put_data_i(.data(weight_abs[4*i+2]), .is_update_w(1'b1), .idx_w(idx_w), .col(2));
					put_data_i(.data(weight_abs[4*i+3]), .is_update_w(1'b1), .idx_w(idx_w), .col(3));
				join
			end
		end

		if(div_step == 4) begin
			for (int i = 0; i < 4; i++) begin
				fork
					put_data_i(.data(weight_abs[2*i+0]), .is_update_w(1'b1), .idx_w(idx_w), .col(0));
					put_data_i(.data(weight_abs[2*i+1]), .is_update_w(1'b1), .idx_w(idx_w), .col(1));
				join
			end
		end

		if(div_step == 8) begin
			for (int i = 0; i < 8; i++) begin
				put_data_i(.data(weight_abs[i]), .is_update_w(1'b1), .idx_w(idx_w), .col(0));
			end
		end

		// update_w   [idx_w] = 1'b0;

	endtask : shift_weight_in

	task automatic shift_acc_in(input int acc_in, input int idx_h, input int idx);
		$display($stime, " shift acc [%-03d][%-03d] <= dx%-04d", idx_h, idx, acc_in);
		for (int i = 0; i < 32; i++) begin
			put_acc_i(.acc_in(acc_in[i]), .idx_h(idx_h), .idx(idx));
		end
	endtask : shift_acc_in

	task automatic shift_mac_en(input int idx_h, input int idx);
		for (int i = 0; i < 32; i++) begin
			put_mac_en(.mac_en_i(1'b1), .idx_h(idx_h), .idx(idx));
		end
	endtask : shift_mac_en

	task automatic shift_clr_and_one_in(input int idx_h, input int idx);
		for (int i = 0; i < 32; i++) begin
			if(i == 0)                    put_clr_and_plus_one(.one(1'b1), .idx_h(idx_h), .idx(idx));
			else                          put_clr_and_plus_one(.one(1'b0), .idx_h(idx_h), .idx(idx));
		end
	endtask : shift_clr_and_one_in

	task automatic get_result(output logic [31:0] result_o, input int idx_h, input int idx);
		for (int i = 0; i < 32; i = ((mac_en_dly_vec[4*idx_h+idx]==1) ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i]), .idx_h(idx_h), .idx(idx));
		end
	endtask : get_result

	task automatic check_32(input logic [31:0] golden, input logic [31:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden %d != got %d", golden, check_result);
			$finish;
		end
	endtask : check_32

	int          check_queue   [4*SUBARRAY_HEIGHT][$];
	int          result_queue  [4*SUBARRAY_HEIGHT][$];
	logic [31:0] shift_acc_in_q[4*SUBARRAY_HEIGHT][$];
	logic [7:0]  shift_dat_in_q[SUBARRAY_WIDTH*NUM_DATAFLOW_PER_MX][$];
	int          shift_wgt_in_q[SUBARRAY_WIDTH][$];



	initial begin: i_run
		automatic int  new_w;
		automatic int  new_data;
		automatic int  new_acc;
		automatic logic [7:0] low_w;
		automatic logic [7:0] low_data;
		automatic int x_group;


		#100;
		get_clk();


	`define FORK_WGT_Q(idx_w_i) \
	if((idx_w_i<SUBARRAY_WIDTH)) begin \
		fork \
			begin \
				automatic int  idx_w = idx_w_i; \
				//$display($stime, " create acc_threding[%d][%d]", idx_h, idx); \
				forever begin \
					if(shift_wgt_in_q[idx_w].size() != 0) begin \
						automatic int new_wgt_pop = shift_wgt_in_q[idx_w].pop_front(); \
						shift_weight_in(.weight(new_wgt_pop), .idx_w(idx_w)); \
					end \
					else begin \
						//@(negedge clk); \
						//get_clk; \
						#1; \
						update_w[idx_w] = 0; \
						//$display($stime, " wait acc_threding[%d][%d]", idx_h, idx); \
					end \
				end \
			end \
		join_none \
	end

	`FORK_WGT_Q(00)
	`FORK_WGT_Q(01)
	`FORK_WGT_Q(02)
	`FORK_WGT_Q(03)
	`FORK_WGT_Q(04)
	`FORK_WGT_Q(05)
	`FORK_WGT_Q(06)
	`FORK_WGT_Q(07)
	`FORK_WGT_Q(08)
	`FORK_WGT_Q(09)
	`FORK_WGT_Q(10)
	`FORK_WGT_Q(11)
	`FORK_WGT_Q(12)
	`FORK_WGT_Q(13)
	`FORK_WGT_Q(14)
	`FORK_WGT_Q(15)


	`define FORK_MAC_Q(idx_h_i, idx_i) \
	if((idx_h_i<SUBARRAY_HEIGHT)) begin \
		fork \
			begin \
				automatic int  idx_h = idx_h_i; \
				automatic int  idx = idx_i; \
				//$display($stime, " create acc_threding[%d][%d]", idx_h, idx); \
				forever begin \
					if(shift_acc_in_q[4*idx_h+idx].size() != 0) begin \
						automatic int new_acc_pop = shift_acc_in_q[4*idx_h+idx].pop_front(); \
						fork \
							shift_acc_in(.acc_in(new_acc_pop), .idx_h(idx_h), .idx(idx)); \
							shift_mac_en(.idx_h(idx_h), .idx(idx)); \
							shift_clr_and_one_in(.idx_h(idx_h), .idx(idx)); \
						join \
					end \
					else begin \
						//@(negedge clk); \
						//get_clk; \
						#1; \
						mac_en[4*idx_h+idx] = 0; \
						//$display($stime, " wait acc_threding[%d][%d]", idx_h, idx); \
					end \
				end \
			end \
		join_none \
	end

	// for (int i = -1; i < 1; i++) begin
	// 	`FORK_MAC_Q(i, 0) `FORK_MAC_Q(i, 1) `FORK_MAC_Q(i, 2) `FORK_MAC_Q(i, 3)
	// 	get_clk;
	// end

		`FORK_MAC_Q(00, 0) `FORK_MAC_Q(00, 1) `FORK_MAC_Q(00, 2) `FORK_MAC_Q(00, 3)
		`FORK_MAC_Q(01, 0) `FORK_MAC_Q(01, 1) `FORK_MAC_Q(01, 2) `FORK_MAC_Q(01, 3)
		`FORK_MAC_Q(02, 0) `FORK_MAC_Q(02, 1) `FORK_MAC_Q(02, 2) `FORK_MAC_Q(02, 3)
		`FORK_MAC_Q(03, 0) `FORK_MAC_Q(03, 1) `FORK_MAC_Q(03, 2) `FORK_MAC_Q(03, 3)
		`FORK_MAC_Q(04, 0) `FORK_MAC_Q(04, 1) `FORK_MAC_Q(04, 2) `FORK_MAC_Q(04, 3)
		`FORK_MAC_Q(05, 0) `FORK_MAC_Q(05, 1) `FORK_MAC_Q(05, 2) `FORK_MAC_Q(05, 3)
		`FORK_MAC_Q(06, 0) `FORK_MAC_Q(06, 1) `FORK_MAC_Q(06, 2) `FORK_MAC_Q(06, 3)
		`FORK_MAC_Q(07, 0) `FORK_MAC_Q(07, 1) `FORK_MAC_Q(07, 2) `FORK_MAC_Q(07, 3)
		`FORK_MAC_Q(08, 0) `FORK_MAC_Q(08, 1) `FORK_MAC_Q(08, 2) `FORK_MAC_Q(08, 3)
		`FORK_MAC_Q(09, 0) `FORK_MAC_Q(09, 1) `FORK_MAC_Q(09, 2) `FORK_MAC_Q(09, 3)
		`FORK_MAC_Q(10, 0) `FORK_MAC_Q(10, 1) `FORK_MAC_Q(10, 2) `FORK_MAC_Q(10, 3)
		`FORK_MAC_Q(11, 0) `FORK_MAC_Q(11, 1) `FORK_MAC_Q(11, 2) `FORK_MAC_Q(11, 3)
		`FORK_MAC_Q(12, 0) `FORK_MAC_Q(12, 1) `FORK_MAC_Q(12, 2) `FORK_MAC_Q(12, 3)
		`FORK_MAC_Q(13, 0) `FORK_MAC_Q(13, 1) `FORK_MAC_Q(13, 2) `FORK_MAC_Q(13, 3)
		`FORK_MAC_Q(14, 0) `FORK_MAC_Q(14, 1) `FORK_MAC_Q(14, 2) `FORK_MAC_Q(14, 3)
		`FORK_MAC_Q(15, 0) `FORK_MAC_Q(15, 1) `FORK_MAC_Q(15, 2) `FORK_MAC_Q(15, 3)


	// fork begin: outter_mac_fork
	// 	for (int i = 0; i < SUBARRAY_HEIGHT; i++) begin: inner_mac_fork
	// 		for (int j = 0; j < 4; j++) begin
	// 			fork
	// 				automatic int idx_h = i;
	// 				automatic int idx   = j;
	// 				automatic int  check_result = 0;
	// 				begin
	// 					forever begin
	// 						if(shift_acc_in_q[idx_h].size() != 0) begin
	// 							automatic int new_acc_pop = shift_acc_in_q[idx_h].pop_front();
	// 							fork
	// 								shift_acc_in(.acc_in(new_acc_pop), .idx_h(idx_h), .idx(idx));
	// 								shift_mac_en(.idx_h(idx_h), .idx(idx));
	// 								shift_clr_and_one_in(.idx_h(idx_h), .idx(idx));
	// 							join
	// 						end
	// 						else begin
	// 							//@(negedge clk);
	// 							//get_clk;
	// 							#1;
	// 							mac_en[4*idx_h+idx] = 0;
	// 						end
	// 					end
	// 				end
	// 			join_none
	// 		end
	// 	end: inner_mac_fork
	// 	wait fork;
	// end: outter_mac_fork
	// join_none


	`define FORK_RESULT_Q(idx_h_i, idx_i) \
	if(idx_h_i<SUBARRAY_HEIGHT) begin \
		fork \
			begin \
				forever begin \
					automatic int  check_result = 0; \
					get_result(check_result, .idx_h(idx_h_i), .idx(idx_i)); \
					$display($stime, " get   res [%-03d][%-03d] <= dx%-04d", idx_h_i, idx_i, check_result); \
					result_queue[4*idx_h_i+idx_i].push_back(check_result); \
				end \
			end \
		join_none \
	end

		`FORK_RESULT_Q(00, 0) `FORK_RESULT_Q(00, 1) `FORK_RESULT_Q(00, 2) `FORK_RESULT_Q(00, 3)
		`FORK_RESULT_Q(01, 0) `FORK_RESULT_Q(01, 1) `FORK_RESULT_Q(01, 2) `FORK_RESULT_Q(01, 3)
		`FORK_RESULT_Q(02, 0) `FORK_RESULT_Q(02, 1) `FORK_RESULT_Q(02, 2) `FORK_RESULT_Q(02, 3)
		`FORK_RESULT_Q(03, 0) `FORK_RESULT_Q(03, 1) `FORK_RESULT_Q(03, 2) `FORK_RESULT_Q(03, 3)
		`FORK_RESULT_Q(04, 0) `FORK_RESULT_Q(04, 1) `FORK_RESULT_Q(04, 2) `FORK_RESULT_Q(04, 3)
		`FORK_RESULT_Q(05, 0) `FORK_RESULT_Q(05, 1) `FORK_RESULT_Q(05, 2) `FORK_RESULT_Q(05, 3)
		`FORK_RESULT_Q(06, 0) `FORK_RESULT_Q(06, 1) `FORK_RESULT_Q(06, 2) `FORK_RESULT_Q(06, 3)
		`FORK_RESULT_Q(07, 0) `FORK_RESULT_Q(07, 1) `FORK_RESULT_Q(07, 2) `FORK_RESULT_Q(07, 3)
		`FORK_RESULT_Q(08, 0) `FORK_RESULT_Q(08, 1) `FORK_RESULT_Q(08, 2) `FORK_RESULT_Q(08, 3)
		`FORK_RESULT_Q(09, 0) `FORK_RESULT_Q(09, 1) `FORK_RESULT_Q(09, 2) `FORK_RESULT_Q(09, 3)
		`FORK_RESULT_Q(10, 0) `FORK_RESULT_Q(10, 1) `FORK_RESULT_Q(10, 2) `FORK_RESULT_Q(10, 3)
		`FORK_RESULT_Q(11, 0) `FORK_RESULT_Q(11, 1) `FORK_RESULT_Q(11, 2) `FORK_RESULT_Q(11, 3)
		`FORK_RESULT_Q(12, 0) `FORK_RESULT_Q(12, 1) `FORK_RESULT_Q(12, 2) `FORK_RESULT_Q(12, 3)
		`FORK_RESULT_Q(13, 0) `FORK_RESULT_Q(13, 1) `FORK_RESULT_Q(13, 2) `FORK_RESULT_Q(13, 3)
		`FORK_RESULT_Q(14, 0) `FORK_RESULT_Q(14, 1) `FORK_RESULT_Q(14, 2) `FORK_RESULT_Q(14, 3)
		`FORK_RESULT_Q(15, 0) `FORK_RESULT_Q(15, 1) `FORK_RESULT_Q(15, 2) `FORK_RESULT_Q(15, 3)

	// fork begin: outter_result_fork
	// 	for (int i = 0; i < SUBARRAY_HEIGHT; i++) begin: inner_result_fork
	// 		for (int j = 0; j < 4; j++) begin
	// 			fork
	// 				automatic int idx_h = i;
	// 				automatic int idx   = j;
	// 				automatic int  check_result = 0;
	// 				begin
	// 					forever begin
	// 						get_result(check_result, .idx_h(idx_h), .idx(idx));
	// 						result_queue[4*idx_h+idx].push_back(check_result);
	// 					end
	// 				end
	// 			join_none
	// 		end
	// 	end: inner_result_fork
	// 	wait fork;
	// end: outter_result_fork
	// join_none

	`define FORK_CHECK_Q(idx_h_i, idx_i) \
	if(idx_h_i<SUBARRAY_HEIGHT) begin \
		fork \
			begin \
				forever begin \
					if((result_queue[4*idx_h_i+idx_i].size() != 0) && (check_queue[4*idx_h_i+idx_i].size() != 0)) begin \
						automatic int pop_result; \
						automatic int pop_check; \
						pop_result = result_queue[4*idx_h_i+idx_i].pop_front(); \
						pop_check  = check_queue [4*idx_h_i+idx_i].pop_front(); \
						$display($stime, " pop check=%-04d, result=%-04d for mac [%-03d][%-03d]", pop_check, pop_result, idx_h_i, idx_i); \
						check_32(pop_check, pop_result); \
					end else begin \
						get_clk; \
					end \
				end \
			end \
		join_none \
	end

		`FORK_CHECK_Q(00, 0) `FORK_CHECK_Q(00, 1) `FORK_CHECK_Q(00, 2) `FORK_CHECK_Q(00, 3)
		`FORK_CHECK_Q(01, 0) `FORK_CHECK_Q(01, 1) `FORK_CHECK_Q(01, 2) `FORK_CHECK_Q(01, 3)
		`FORK_CHECK_Q(02, 0) `FORK_CHECK_Q(02, 1) `FORK_CHECK_Q(02, 2) `FORK_CHECK_Q(02, 3)
		`FORK_CHECK_Q(03, 0) `FORK_CHECK_Q(03, 1) `FORK_CHECK_Q(03, 2) `FORK_CHECK_Q(03, 3)
		`FORK_CHECK_Q(04, 0) `FORK_CHECK_Q(04, 1) `FORK_CHECK_Q(04, 2) `FORK_CHECK_Q(04, 3)
		`FORK_CHECK_Q(05, 0) `FORK_CHECK_Q(05, 1) `FORK_CHECK_Q(05, 2) `FORK_CHECK_Q(05, 3)
		`FORK_CHECK_Q(06, 0) `FORK_CHECK_Q(06, 1) `FORK_CHECK_Q(06, 2) `FORK_CHECK_Q(06, 3)
		`FORK_CHECK_Q(07, 0) `FORK_CHECK_Q(07, 1) `FORK_CHECK_Q(07, 2) `FORK_CHECK_Q(07, 3)
		`FORK_CHECK_Q(08, 0) `FORK_CHECK_Q(08, 1) `FORK_CHECK_Q(08, 2) `FORK_CHECK_Q(08, 3)
		`FORK_CHECK_Q(09, 0) `FORK_CHECK_Q(09, 1) `FORK_CHECK_Q(09, 2) `FORK_CHECK_Q(09, 3)
		`FORK_CHECK_Q(10, 0) `FORK_CHECK_Q(10, 1) `FORK_CHECK_Q(10, 2) `FORK_CHECK_Q(10, 3)
		`FORK_CHECK_Q(11, 0) `FORK_CHECK_Q(11, 1) `FORK_CHECK_Q(11, 2) `FORK_CHECK_Q(11, 3)
		`FORK_CHECK_Q(12, 0) `FORK_CHECK_Q(12, 1) `FORK_CHECK_Q(12, 2) `FORK_CHECK_Q(12, 3)
		`FORK_CHECK_Q(13, 0) `FORK_CHECK_Q(13, 1) `FORK_CHECK_Q(13, 2) `FORK_CHECK_Q(13, 3)
		`FORK_CHECK_Q(14, 0) `FORK_CHECK_Q(14, 1) `FORK_CHECK_Q(14, 2) `FORK_CHECK_Q(14, 3)
		`FORK_CHECK_Q(15, 0) `FORK_CHECK_Q(15, 1) `FORK_CHECK_Q(15, 2) `FORK_CHECK_Q(15, 3)

	// fork begin: outter_check_fork
	// 	for (int i = 0; i < SUBARRAY_HEIGHT; i++) begin: inner_check_fork
	// 		for (int j = 0; j < 4; j++) begin
	// 			fork
	// 				automatic int idx_h = i;
	// 				automatic int idx   = j;
	// 				automatic int  check_result = 0;
	// 				begin
	// 					forever begin
	// 						if((result_queue[4*idx_h+idx].size() != 0) && (check_queue[4*idx_h+idx].size() != 0)) begin
	// 							automatic int pop_result;
	// 							automatic int pop_check;
	// 							pop_result = result_queue[4*idx_h+idx].pop_front();
	// 							pop_check  = check_queue [4*idx_h+idx].pop_front();
	// 							$display($stime, " pop check=%d, result=%d for mac_%d", pop_check, pop_result, 4*idx_h+idx);
	// 							check_32(pop_check, pop_result);
	// 						end else begin
	// 							get_clk;
	// 						end
	// 					end
	// 				end
	// 			join_none
	// 		end
	// 	end: inner_check_fork
	// 	wait fork;
	// end: outter_check_fork
	// join_none


	`define FORK_DATA_Q(idx_w_i, idx_mux_i) \
	if((idx_w_i<SUBARRAY_WIDTH) && (idx_mux_i<W_DATA_MUX)) begin \
		fork \
			begin \
				forever begin \
					if(shift_dat_in_q[idx_w_i*NUM_DATAFLOW_PER_MX+idx_mux_i].size() != 0) begin \
						automatic logic [7:0] new_dat_pop = shift_dat_in_q[idx_w_i*NUM_DATAFLOW_PER_MX+idx_mux_i].pop_front(); \
						$display($stime, " shift dat [%-03d][%-03d] <= dx%-04d", idx_w_i, idx_mux_i, new_dat_pop); \
						shift_data_in(.data(new_dat_pop), .idx_w(idx_w_i), .col(idx_mux_i)); \
					end \
					else begin \
						#1; \
					end \
				end \
			end \
		join_none \
	end

	`FORK_DATA_Q(00, 0) `FORK_DATA_Q(00, 1) `FORK_DATA_Q(00, 2) `FORK_DATA_Q(00, 3) `FORK_DATA_Q(00, 4) `FORK_DATA_Q(00, 5) `FORK_DATA_Q(00, 6) `FORK_DATA_Q(00, 7) 
	`FORK_DATA_Q(01, 0) `FORK_DATA_Q(01, 1) `FORK_DATA_Q(01, 2) `FORK_DATA_Q(01, 3) `FORK_DATA_Q(01, 4) `FORK_DATA_Q(01, 5) `FORK_DATA_Q(01, 6) `FORK_DATA_Q(01, 7) 
	`FORK_DATA_Q(02, 0) `FORK_DATA_Q(02, 1) `FORK_DATA_Q(02, 2) `FORK_DATA_Q(02, 3) `FORK_DATA_Q(02, 4) `FORK_DATA_Q(02, 5) `FORK_DATA_Q(02, 6) `FORK_DATA_Q(02, 7) 
	`FORK_DATA_Q(03, 0) `FORK_DATA_Q(03, 1) `FORK_DATA_Q(03, 2) `FORK_DATA_Q(03, 3) `FORK_DATA_Q(03, 4) `FORK_DATA_Q(03, 5) `FORK_DATA_Q(03, 6) `FORK_DATA_Q(03, 7) 
	`FORK_DATA_Q(04, 0) `FORK_DATA_Q(04, 1) `FORK_DATA_Q(04, 2) `FORK_DATA_Q(04, 3) `FORK_DATA_Q(04, 4) `FORK_DATA_Q(04, 5) `FORK_DATA_Q(04, 6) `FORK_DATA_Q(04, 7) 
	`FORK_DATA_Q(05, 0) `FORK_DATA_Q(05, 1) `FORK_DATA_Q(05, 2) `FORK_DATA_Q(05, 3) `FORK_DATA_Q(05, 4) `FORK_DATA_Q(05, 5) `FORK_DATA_Q(05, 6) `FORK_DATA_Q(05, 7) 
	`FORK_DATA_Q(06, 0) `FORK_DATA_Q(06, 1) `FORK_DATA_Q(06, 2) `FORK_DATA_Q(06, 3) `FORK_DATA_Q(06, 4) `FORK_DATA_Q(06, 5) `FORK_DATA_Q(06, 6) `FORK_DATA_Q(06, 7) 
	`FORK_DATA_Q(07, 0) `FORK_DATA_Q(07, 1) `FORK_DATA_Q(07, 2) `FORK_DATA_Q(07, 3) `FORK_DATA_Q(07, 4) `FORK_DATA_Q(07, 5) `FORK_DATA_Q(07, 6) `FORK_DATA_Q(07, 7) 
	`FORK_DATA_Q(08, 0) `FORK_DATA_Q(08, 1) `FORK_DATA_Q(08, 2) `FORK_DATA_Q(08, 3) `FORK_DATA_Q(08, 4) `FORK_DATA_Q(08, 5) `FORK_DATA_Q(08, 6) `FORK_DATA_Q(08, 7) 
	`FORK_DATA_Q(09, 0) `FORK_DATA_Q(09, 1) `FORK_DATA_Q(09, 2) `FORK_DATA_Q(09, 3) `FORK_DATA_Q(09, 4) `FORK_DATA_Q(09, 5) `FORK_DATA_Q(09, 6) `FORK_DATA_Q(09, 7) 
	`FORK_DATA_Q(10, 0) `FORK_DATA_Q(10, 1) `FORK_DATA_Q(10, 2) `FORK_DATA_Q(10, 3) `FORK_DATA_Q(10, 4) `FORK_DATA_Q(10, 5) `FORK_DATA_Q(10, 6) `FORK_DATA_Q(10, 7) 
	`FORK_DATA_Q(11, 0) `FORK_DATA_Q(11, 1) `FORK_DATA_Q(11, 2) `FORK_DATA_Q(11, 3) `FORK_DATA_Q(11, 4) `FORK_DATA_Q(11, 5) `FORK_DATA_Q(11, 6) `FORK_DATA_Q(11, 7) 
	`FORK_DATA_Q(12, 0) `FORK_DATA_Q(12, 1) `FORK_DATA_Q(12, 2) `FORK_DATA_Q(12, 3) `FORK_DATA_Q(12, 4) `FORK_DATA_Q(12, 5) `FORK_DATA_Q(12, 6) `FORK_DATA_Q(12, 7) 
	`FORK_DATA_Q(13, 0) `FORK_DATA_Q(13, 1) `FORK_DATA_Q(13, 2) `FORK_DATA_Q(13, 3) `FORK_DATA_Q(13, 4) `FORK_DATA_Q(13, 5) `FORK_DATA_Q(13, 6) `FORK_DATA_Q(13, 7) 
	`FORK_DATA_Q(14, 0) `FORK_DATA_Q(14, 1) `FORK_DATA_Q(14, 2) `FORK_DATA_Q(14, 3) `FORK_DATA_Q(14, 4) `FORK_DATA_Q(14, 5) `FORK_DATA_Q(14, 6) `FORK_DATA_Q(14, 7) 
	`FORK_DATA_Q(15, 0) `FORK_DATA_Q(15, 1) `FORK_DATA_Q(15, 2) `FORK_DATA_Q(15, 3) `FORK_DATA_Q(15, 4) `FORK_DATA_Q(15, 5) `FORK_DATA_Q(15, 6) `FORK_DATA_Q(15, 7) 


	// fork begin: outter_data_fork
	// 	for (int i = 0; i < SUBARRAY_WIDTH; i++) begin: inner_data_fork
	// 		for (int j = 0; j < W_DATA_MUX; j++) begin
	// 			fork
	// 				automatic int idx_w   = i;
	// 				automatic int idx_mux = j;
	// 				begin
	// 					forever begin
	// 						if(shift_dat_in_q[idx_w][idx_mux].size() != 0) begin
	// 							automatic logic [7:0] new_dat_pop = shift_dat_in_q[idx_w][idx_mux].pop_front();
	// 							shift_data_in(.data(new_dat_pop), .idx_w(idx_w), .col(idx_mux));
	// 						end
	// 						else begin
	// 							//@(negedge clk);
	// 							//get_clk;
	// 							#1;
	// 						end
	// 					end
	// 				end
	// 			join_none
	// 		end
	// 	end: inner_data_fork
	// 	wait fork;
	// end: outter_data_fork
	// join_none

		while(|mac_en) get_clk;

		new_w = -128;
		for (int j = 0; j < SUBARRAY_HEIGHT; j++) begin
			for (int i = 0; i < SUBARRAY_WIDTH; i++) begin
				automatic logic[7:0] weight;
				automatic int        abs_new_w;
				cur_weight[i][j] = new_w;
				weight = new_w;
				put_control1(.control1_i( (new_w>=0) ? 1'b1 : 1'b0), .idx_w(i), .idx_h(j));
				new_w ++;
				abs_new_w = (new_w > 0) ? new_w : -new_w;
				if(abs_new_w>255)
					new_w = 0;
			end
		end

		new_acc = -1024;
		for (int j = 0; j < SUBARRAY_HEIGHT; j++) begin
			for (int i = 0; i < 4; i++) begin
				cur_acc[j][i] = new_acc;
				new_acc ++;
			end
		end

		new_data = 1;
		for (int i = 0; i < (4*SUBARRAY_HEIGHT); i++) begin
			for (int j = 0; j < SUBARRAY_WIDTH; j++) begin
				cur_dat[j][0][i] = new_data;
				new_data ++;
				if(new_data>255)
					new_data = 0;
			end
		end		

		for (int i = 0; i < SUBARRAY_WIDTH; i++) begin
			for (int j=(SUBARRAY_HEIGHT-1); j>=0; j--) begin
					automatic int idx_w   = i;
					automatic int idx_h   = j;
					$display($stime, " shift wgt [%-03d][%-03d] <= 0x%-04d", idx_w, idx_h, cur_weight[idx_w][idx_h]);
					// shift_weight_in(.weight(cur_weight[idx_w][idx_h]), .idx_w(idx_w));
					shift_wgt_in_q[idx_w].push_back(cur_weight[idx_w][idx_h]);
			end
		end

		repeat((8/NUM_DATAFLOW_PER_MX)*(SUBARRAY_HEIGHT)+1) get_clk;


		`define ISSUE_DAT(cyc) \
			begin \
				if(cyc<SUBARRAY_WIDTH) begin \
					repeat(cyc) get_clk; \
					shift_dat_in_q[cyc*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[cyc][0][0]); \
					shift_dat_in_q[cyc*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[cyc][0][1]); \
					shift_dat_in_q[cyc*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[cyc][0][2]); \
					shift_dat_in_q[cyc*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[cyc][0][3]); \
				end \
			end

		`define ISSUE_ACC(idx_h, idx) \
			begin \
				if(idx_h<SUBARRAY_HEIGHT) \
					begin repeat(idx_h+8*idx) get_clk; shift_acc_in_q[4*idx_h+idx].push_back(cur_acc[idx_h][idx]); end \
			end

		fork
			// begin repeat(00) get_clk; 
			// 	                shift_dat_in_q[0*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[0][0][0]); 
			//                  shift_dat_in_q[0*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[0][0][1]); 
			//                  shift_dat_in_q[0*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[0][0][2]); 
			//                  shift_dat_in_q[0*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[0][0][3]); 
			// end
			// begin repeat(01) get_clk; 
			// 	                shift_dat_in_q[1*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[1][0][0]);
			//                  shift_dat_in_q[1*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[1][0][1]);
			//                  shift_dat_in_q[1*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[1][0][2]);
			//                  shift_dat_in_q[1*NUM_DATAFLOW_PER_MX+0].push_back(cur_dat[1][0][3]); 
			// end

			`ISSUE_DAT(00)
			`ISSUE_DAT(01)
			`ISSUE_DAT(02)
			`ISSUE_DAT(03)
			`ISSUE_DAT(04)
			`ISSUE_DAT(05)
			`ISSUE_DAT(06)
			`ISSUE_DAT(07)
			`ISSUE_DAT(08)
			`ISSUE_DAT(09)
			`ISSUE_DAT(10)
			`ISSUE_DAT(11)
			`ISSUE_DAT(12)
			`ISSUE_DAT(13)
			`ISSUE_DAT(14)
			`ISSUE_DAT(15)


			`ISSUE_ACC(00, 0) `ISSUE_ACC(00, 1) `ISSUE_ACC(00, 2) `ISSUE_ACC(00, 3)
			`ISSUE_ACC(01, 0) `ISSUE_ACC(01, 1) `ISSUE_ACC(01, 2) `ISSUE_ACC(01, 3)
			`ISSUE_ACC(02, 0) `ISSUE_ACC(02, 1) `ISSUE_ACC(02, 2) `ISSUE_ACC(02, 3)
			`ISSUE_ACC(03, 0) `ISSUE_ACC(03, 1) `ISSUE_ACC(03, 2) `ISSUE_ACC(03, 3)
			`ISSUE_ACC(04, 0) `ISSUE_ACC(04, 1) `ISSUE_ACC(04, 2) `ISSUE_ACC(04, 3)
			`ISSUE_ACC(05, 0) `ISSUE_ACC(05, 1) `ISSUE_ACC(05, 2) `ISSUE_ACC(05, 3)
			`ISSUE_ACC(06, 0) `ISSUE_ACC(06, 1) `ISSUE_ACC(06, 2) `ISSUE_ACC(06, 3)
			`ISSUE_ACC(07, 0) `ISSUE_ACC(07, 1) `ISSUE_ACC(07, 2) `ISSUE_ACC(07, 3)
			`ISSUE_ACC(08, 0) `ISSUE_ACC(08, 1) `ISSUE_ACC(08, 2) `ISSUE_ACC(08, 3)
			`ISSUE_ACC(09, 0) `ISSUE_ACC(09, 1) `ISSUE_ACC(09, 2) `ISSUE_ACC(09, 3)
			`ISSUE_ACC(10, 0) `ISSUE_ACC(10, 1) `ISSUE_ACC(10, 2) `ISSUE_ACC(10, 3)
			`ISSUE_ACC(11, 0) `ISSUE_ACC(11, 1) `ISSUE_ACC(11, 2) `ISSUE_ACC(11, 3)
			`ISSUE_ACC(12, 0) `ISSUE_ACC(12, 1) `ISSUE_ACC(12, 2) `ISSUE_ACC(12, 3)
			`ISSUE_ACC(13, 0) `ISSUE_ACC(13, 1) `ISSUE_ACC(13, 2) `ISSUE_ACC(13, 3)
			`ISSUE_ACC(14, 0) `ISSUE_ACC(14, 1) `ISSUE_ACC(14, 2) `ISSUE_ACC(14, 3)
			`ISSUE_ACC(15, 0) `ISSUE_ACC(15, 1) `ISSUE_ACC(15, 2) `ISSUE_ACC(15, 3)
			// begin repeat(00) get_clk; shift_acc_in_q[0]   .push_back(cur_acc[0][0]); end
			// begin repeat(08) get_clk; shift_acc_in_q[1]   .push_back(cur_acc[0][1]); end
			// begin repeat(16) get_clk; shift_acc_in_q[2]   .push_back(cur_acc[0][2]); end
			// begin repeat(24) get_clk; shift_acc_in_q[3]   .push_back(cur_acc[0][3]); end
			// begin repeat(01) get_clk; shift_acc_in_q[4]   .push_back(cur_acc[1][0]); end
			// begin repeat(09) get_clk; shift_acc_in_q[5]   .push_back(cur_acc[1][1]); end
			// begin repeat(17) get_clk; shift_acc_in_q[6]   .push_back(cur_acc[1][2]); end
			// begin repeat(25) get_clk; shift_acc_in_q[7]   .push_back(cur_acc[1][3]); end
		join

		x_group = 0;
		for (int j = 0; j < (4*SUBARRAY_HEIGHT); j++) begin
			automatic int check_result;
			automatic int new_data;
			check_result = cur_acc[j/4][j%4];
			if((j%4) == 0) x_group = 0;

			for (int i = 0; i < SUBARRAY_WIDTH; i++) begin
				new_data = {24'b0, cur_dat[i][0][x_group]};
				check_result += cur_weight[i][j/4]*new_data;
			end
			x_group ++;
			check_queue[j].push_back(check_result);
		end

			// for(new_acc=-10; new_acc<10; new_acc++) begin
			// 	for(new_data=0; new_data<10; new_data++) begin
					
			// 		low_data = new_data;
			// 		//$display($stime, " low_w=0xb%b, low_data=0xb%b, new_w=%d, new_data=%d", low_w, low_data, new_w, new_data);
			// 		//fork
			// 		//	begin
			// 				//get_clk(); // wait clear acc
			// 				fork
			// 					shift_acc_in_q[0].push_back(new_acc+0);
			// 					shift_data_in(.data(low_data+0), .col(0));
			// 				join

			// 				fork
			// 					shift_acc_in_q[1].push_back(new_acc+1);
			// 					shift_data_in(.data(low_data+1), .col(0));
			// 				join
							
			// 				fork
			// 					shift_acc_in_q[2].push_back(new_acc+2);
			// 					shift_data_in(.data(low_data+2), .col(0));
			// 				join

			// 				fork
			// 					shift_acc_in_q[3].push_back(new_acc+3);
			// 					shift_data_in(.data(low_data+3), .col(0));
			// 				join
			// 		//	end

			// 			//begin repeat(00) get_clk; shift_mac_en(.idx(0)); end
			// 			//begin repeat(08) get_clk; shift_mac_en(.idx(1)); end
			// 			//begin repeat(16) get_clk; shift_mac_en(.idx(2)); end
			// 			//begin repeat(24) get_clk; shift_mac_en(.idx(3)); end

			// 			// begin //repeat(00) get_clk;/*shift_acc_in(.acc_in(new_acc+0), .idx(0));*/
							
			// 			// end
			// 			// begin repeat(08) get_clk;/*shift_acc_in(.acc_in(new_acc+1), .idx(1));*/
			// 			// 	shift_acc_in_q[1].push_back(new_acc+1);
			// 			// end
			// 			// begin repeat(16) get_clk;/*shift_acc_in(.acc_in(new_acc+2), .idx(2));*/
			// 			// 	shift_acc_in_q[2].push_back(new_acc+2);
			// 			// end
			// 			// begin repeat(24) get_clk;/*shift_acc_in(.acc_in(new_acc+3), .idx(3));*/
			// 			// 	shift_acc_in_q[3].push_back(new_acc+3);
			// 			// end
												
			// 			//begin repeat(00) get_clk; shift_clr_and_one_in(.idx(0)); end
			// 			//begin repeat(08) get_clk; shift_clr_and_one_in(.idx(1)); end
			// 			//begin repeat(16) get_clk; shift_clr_and_one_in(.idx(2)); end
			// 			//begin repeat(24) get_clk; shift_clr_and_one_in(.idx(3)); end

			// 		//join

		// 			for (int idx = 0; idx < 4; idx++) begin
		// 				golden_mac = (new_data+idx)*(new_w) + (new_acc+idx);
		// 				$display(" (%4d * %4d) + %4d = %4d for mac_%d", new_data+idx, new_w, new_acc+idx, golden_mac, idx);
		// 				check_queue[idx].push_back(golden_mac);
		// 			end
		// 		end
		// 	end
		// end

		//repeat(100) get_clk();

		// for (int i = 0; i < count; i++) begin
		// 	/* code */
		// end


		foreach(check_queue[idx]) begin
			while(check_queue[idx].size()!=0) get_clk;
		end

		// while(check_queue[0].size()!=0) get_clk;
		// while(check_queue[1].size()!=0) get_clk;
		// while(check_queue[2].size()!=0) get_clk;
		// while(check_queue[3].size()!=0) get_clk;
		get_clk;
		$finish;
	end

endmodule