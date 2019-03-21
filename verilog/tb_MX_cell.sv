module tb_MX_cell;


	logic       clk                                ;
	logic       reset                              ;
	logic       dataflow_select         = 'b0      ;
	logic [3:0] accumulation            = 'b0      ;
	logic       plus_one                = 'b0      ;
	logic       clear_accu_control      = 'b0      ;
	logic [1:0] dataflow_in             = 'b0      ;
	logic       control1                = 'b0      ;
	logic       update_w                = 'b0      ;
	logic [3:0] mac_en                  = 'b0      ;
	wire  [3:0] result                             ;
	wire  [1:0] input_accu_adder                   ;
	logic [3:0] clr_and_plus_one        = 'b0      ;
	reg   [7:0] cur_weight                         ;
	int         time_cnt          [3:0] = {0,0,0,0};

	wire [31:0] time_cnt_0;
	wire [31:0] time_cnt_1;
	wire [31:0] time_cnt_2;
	wire [31:0] time_cnt_3;

	assign time_cnt_0 = time_cnt[0];
	assign time_cnt_1 = time_cnt[1];
	assign time_cnt_2 = time_cnt[2];
	assign time_cnt_3 = time_cnt[3];

	wire [1:0] dataflow_out      ;
	reg  [3:0] clr_and_plus_one_o;
	wire [3:0] mac_en_o          ;

	j_MX_cell i_j_MX_cell (
		.clk               (clk               ),
		.dataflow_in       (dataflow_in       ), // TODO: Check connection ! Signal/port not matching : Expecting logic [1:0]  -- Found logic
		.dataflow_select   (dataflow_select   ),
		.update_w_i        (update_w          ),
		.reset             (reset             ),
		.control1          (control1          ), // TODO: Check connection ! Signal/port not matching : Expecting logic [3:0]  -- Found logic
		.clr_and_plus_one_i(clr_and_plus_one  ),
		.mac_en_i          (mac_en            ),
		.result            (result            ), // TODO: Check connection ! Signal/port not matching : Expecting logic [3:0]  -- Found logic
		.dataflow_out      (dataflow_out      ),
		.clr_and_plus_one_o(clr_and_plus_one_o),
		.mac_en_o          (mac_en_o          ),
		.accumulation_in   (accumulation      )
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

	// task automatic put_clear_acc();
	// 	clear_accu_control = 1'b1;
	// 	@(posedge clk);
	// 	clear_accu_control = 1'b0;
	// endtask : put_clear_acc	

	task automatic put_data_i(input data, input is_update_w, input int col);
		update_w    = is_update_w;
		dataflow_in[col] = data;
		@(posedge clk);
	endtask : put_data_i

	task automatic put_acc_i(input acc_in, int idx);
		accumulation[idx]    = acc_in;
		@(posedge clk);
	endtask : put_acc_i	

	task automatic put_clr_and_plus_one(input one, input int idx);
		clr_and_plus_one[idx]    = one;
		@(posedge clk);
	endtask : put_clr_and_plus_one

	task automatic put_control1(input control1_i);
		control1    = control1_i;
	endtask : put_control1

	task automatic get_result_bit(output logic result_o, input int idx);
		@(posedge clk);
		result_o = result[idx];
		//$display($stime, "timer = %d, result_o = %b", time_cnt, result_o);
	endtask : get_result_bit

	task automatic put_mac_en(input mac_en_i, input int idx);
		mac_en[idx] = mac_en_i;
		@(posedge clk);
	endtask : put_mac_en


	// reg [1:0] mac_en_dly [3:0];
	// wire mac_en_dly_0 = mac_en_dly[0][0];
	// wire mac_en_dly_1 = mac_en_dly[1][0];
	// wire mac_en_dly_2 = mac_en_dly[2][0];
	// wire mac_en_dly_3 = mac_en_dly[3][0];

	wire [3:0] mac_en_dly_vec = {mac_en_o};

	// always @(posedge clk) begin : proc_mac_en_dly
	// 	if(reset) begin
	// 		mac_en_dly[0] <= 0;
	// 		mac_en_dly[1] <= 0;
	// 		mac_en_dly[2] <= 0;
	// 		mac_en_dly[3] <= 0;
	// 	end else begin
	// 		mac_en_dly[0] <= {mac_en_dly[0][0], mac_en[0]};
	// 		mac_en_dly[1] <= {mac_en_dly[1][0], mac_en[1]};
	// 		mac_en_dly[2] <= {mac_en_dly[2][0], mac_en[2]};
	// 		mac_en_dly[3] <= {mac_en_dly[3][0], mac_en[3]};
	// 	end
	// end



	// parallel part
	task automatic shift_data_in(input logic [7:0] data, input int col);
		//get_clk;
		for (int i = 0; i < 8; i++) begin
			put_data_i(.data(data[i]), .is_update_w(1'b0), .col(col));
		end
	endtask : shift_data_in

	task automatic shift_weight_in(input int weight);
		reg [7:0] weight_abs;
		weight_abs = weight < 0 ? -weight : weight;
		put_control1(.control1_i(weight>=0 ? 1'b1 : 1'b0));
		for (int i = 0; i < 4; i++) begin
			fork
				put_data_i(.data(weight_abs[2*i+0]), .is_update_w(1'b1), .col(0));
				put_data_i(.data(weight_abs[2*i+1]), .is_update_w(1'b1), .col(1));
			join
		end
	endtask : shift_weight_in

	task automatic shift_acc_in(input logic [31:0] acc_in, input int idx);
		for (int i = 0; i < 32; i++) begin
			//$display($stime, " shift acc %d bit[%d] <= %b", idx, i, acc_in[i]);
			put_acc_i(.acc_in(acc_in[i]), .idx(idx));
		end
	endtask : shift_acc_in

	task automatic shift_mac_en(input int idx);
		time_cnt[idx] = 0;
		for (int i = 0; i < 32; i++) begin
			put_mac_en(.mac_en_i(1'b1), .idx(idx));
			time_cnt[idx] = time_cnt[idx] + 1;
		end
		// put_mac_en(.mac_en_i(1'b0));
		// mac_en[idx] = 1'b0;
	endtask : shift_mac_en

	task automatic shift_clr_and_one_in(input int idx);
		//put_clr_and_plus_one(.one(1'b1), .idx(idx));
		for (int i = 0; i < 32; i++) begin
			if(i == 0)                    put_clr_and_plus_one(.one(1'b1), .idx(idx));
			else                          put_clr_and_plus_one(.one(1'b0), .idx(idx));
		end
	endtask : shift_clr_and_one_in

	task automatic get_result(output logic [31:0] result_o, input int idx);
		for (int i = 0; i < 32; i = ((mac_en_dly_vec[idx]==1) ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i]), .idx(idx));
			//if(idx==0) $display($stime, " get result %d bit[%d] = %b", idx, i, result_o[i]);
		end
	endtask : get_result

	task automatic check_32(input logic [31:0] golden, input logic [31:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden %d != got %d", golden, check_result);
			$finish;
		end
	endtask : check_32


	int check_queue[4][$];
	int result_queue[4][$];
	logic [31:0] shift_acc_in_q[4][$];


	initial begin: i_run
		automatic int  check_result[3:0] = {0, 0, 0, 0};
		automatic int  new_w;
		automatic int  new_data;
		automatic int  new_acc;
		automatic logic [7:0] low_w;
		automatic logic [7:0] low_data;

		#100;
		get_clk();


		fork
			begin
				forever begin
					if(shift_acc_in_q[0].size() != 0) begin
						automatic int new_acc_pop = shift_acc_in_q[0].pop_front();
						fork
							shift_acc_in(.acc_in(new_acc_pop), .idx(0));
							shift_mac_en(.idx(0));
							shift_clr_and_one_in(.idx(0));
						join
					end
					else begin
						//@(negedge clk);
						//get_clk;
						#1;
						mac_en[0] = 0;
					end
				end
			end

			begin
				forever begin
					if(shift_acc_in_q[1].size() != 0) begin
						automatic int new_acc_pop = shift_acc_in_q[1].pop_front();
						fork
							shift_acc_in(.acc_in(new_acc_pop), .idx(1));
							shift_mac_en(.idx(1));
							shift_clr_and_one_in(.idx(1));
						join
					end
					else begin
						//@(negedge clk);
						//get_clk;
						#1;
						mac_en[1] = 0;
					end
				end
			end

			begin
				forever begin
					if(shift_acc_in_q[2].size() != 0) begin
						automatic int new_acc_pop = shift_acc_in_q[2].pop_front();
						fork
							shift_acc_in(.acc_in(new_acc_pop), .idx(2));
							shift_mac_en(.idx(2));
							shift_clr_and_one_in(.idx(2));
						join
					end
					else begin
						//@(negedge clk);
						//get_clk;
						#1;
						mac_en[2] = 0;
					end
				end
			end

			begin
				forever begin
					if(shift_acc_in_q[3].size() != 0) begin
						automatic int new_acc_pop = shift_acc_in_q[3].pop_front();
						fork
							shift_acc_in(.acc_in(new_acc_pop), .idx(3));
							shift_mac_en(.idx(3));
							shift_clr_and_one_in(.idx(3));
						join
					end
					else begin
						//@(negedge clk);
						//get_clk;
						#1;
						mac_en[3] = 0;
					end
				end
			end

			begin
				forever begin
					get_result(check_result[0], .idx(0));
					result_queue[0].push_back(check_result[0]);
				end
			end

			begin
				forever begin
					get_result(check_result[1], .idx(1));
					result_queue[1].push_back(check_result[1]);
				end
			end

			begin
				forever begin
					get_result(check_result[2], .idx(2));
					result_queue[2].push_back(check_result[2]);
				end
			end

			begin
				forever begin
					get_result(check_result[3], .idx(3));
					result_queue[3].push_back(check_result[3]);
				end
			end


			begin
				forever begin
					if((result_queue[0].size() != 0) && (check_queue[0].size() != 0)) begin
						automatic int pop_result;
						automatic int pop_check;
						pop_result = result_queue[0].pop_front();
						pop_check  = check_queue [0].pop_front();
						$display($stime, " pop check=%d, result=%d for mac_0", pop_check, pop_result);
						check_32(pop_check, pop_result);
					end else begin
						get_clk;
					end
				end
			end

			begin
				forever begin
					if((result_queue[1].size() != 0) && (check_queue[1].size() != 0)) begin
						automatic int pop_result;
						automatic int pop_check;
						pop_result = result_queue[1].pop_front();
						pop_check  = check_queue [1].pop_front();
						$display($stime, " pop check=%d, result=%d  for mac_1", pop_check, pop_result);
						check_32(pop_check, pop_result);
					end else begin
						get_clk;
					end
				end
			end

			begin
				forever begin
					if((result_queue[2].size() != 0) && (check_queue[2].size() != 0)) begin
						automatic int pop_result;
						automatic int pop_check;
						pop_result = result_queue[2].pop_front();
						pop_check  = check_queue [2].pop_front();
						$display($stime, " pop check=%d, result=%d  for mac_2", pop_check, pop_result);
						check_32(pop_check, pop_result);
					end else begin
						get_clk;
					end
				end
			end

			begin
				forever begin
					if((result_queue[3].size() != 0) && (check_queue[3].size() != 0)) begin
						automatic int pop_result;
						automatic int pop_check;
						pop_result = result_queue[3].pop_front();
						pop_check  = check_queue [3].pop_front();
						$display($stime, " pop check=%d, result=%d  for mac_3", pop_check, pop_result);
						check_32(pop_check, pop_result);
					end else begin
						get_clk;
					end
				end
			end

		join_none


		for(new_w=-10; new_w<10; new_w++) begin
			int golden_mac;
			// low_w = new_w;
			// fork
			// 	put_mac_en(.mac_en_i(1'b0), .idx(0));
			// 	put_mac_en(.mac_en_i(1'b0), .idx(1));
			// 	put_mac_en(.mac_en_i(1'b0), .idx(2));
			// 	put_mac_en(.mac_en_i(1'b0), .idx(3));	
			// join
			while(mac_en[3:0] != 4'b0000) get_clk;
			
			shift_weight_in(.weight(new_w));
			for(new_acc=-10; new_acc<10; new_acc++) begin
				for(new_data=0; new_data<10; new_data++) begin
					
					low_data = new_data;
					//$display($stime, " low_w=0xb%b, low_data=0xb%b, new_w=%d, new_data=%d", low_w, low_data, new_w, new_data);
					//fork
					//	begin
							//get_clk(); // wait clear acc
							fork
								shift_acc_in_q[0].push_back(new_acc+0);
								shift_data_in(.data(low_data+0), .col(0));
							join

							fork
								shift_acc_in_q[1].push_back(new_acc+1);
								shift_data_in(.data(low_data+1), .col(0));
							join
							
							fork
								shift_acc_in_q[2].push_back(new_acc+2);
								shift_data_in(.data(low_data+2), .col(0));
							join

							fork
								shift_acc_in_q[3].push_back(new_acc+3);
								shift_data_in(.data(low_data+3), .col(0));
							join
					//	end

						//begin repeat(00) get_clk; shift_mac_en(.idx(0)); end
						//begin repeat(08) get_clk; shift_mac_en(.idx(1)); end
						//begin repeat(16) get_clk; shift_mac_en(.idx(2)); end
						//begin repeat(24) get_clk; shift_mac_en(.idx(3)); end

						// begin //repeat(00) get_clk;/*shift_acc_in(.acc_in(new_acc+0), .idx(0));*/
							
						// end
						// begin repeat(08) get_clk;/*shift_acc_in(.acc_in(new_acc+1), .idx(1));*/
						// 	shift_acc_in_q[1].push_back(new_acc+1);
						// end
						// begin repeat(16) get_clk;/*shift_acc_in(.acc_in(new_acc+2), .idx(2));*/
						// 	shift_acc_in_q[2].push_back(new_acc+2);
						// end
						// begin repeat(24) get_clk;/*shift_acc_in(.acc_in(new_acc+3), .idx(3));*/
						// 	shift_acc_in_q[3].push_back(new_acc+3);
						// end
												
						//begin repeat(00) get_clk; shift_clr_and_one_in(.idx(0)); end
						//begin repeat(08) get_clk; shift_clr_and_one_in(.idx(1)); end
						//begin repeat(16) get_clk; shift_clr_and_one_in(.idx(2)); end
						//begin repeat(24) get_clk; shift_clr_and_one_in(.idx(3)); end

					//join

					for (int idx = 0; idx < 4; idx++) begin
						golden_mac = (new_data+idx)*(new_w) + (new_acc+idx);
						$display(" (%4d * %4d) + %4d = %4d for mac_%d", new_data+idx, new_w, new_acc+idx, golden_mac, idx);
						check_queue[idx].push_back(golden_mac);
					end
				end
			end
		end

		get_clk();
		while(check_queue[0].size()!=0) get_clk;
		while(check_queue[1].size()!=0) get_clk;
		while(check_queue[2].size()!=0) get_clk;
		while(check_queue[3].size()!=0) get_clk;
		get_clk;
		$finish;
	end

endmodule