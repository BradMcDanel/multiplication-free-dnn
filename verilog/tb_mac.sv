module tb_mac;


	logic       clk                     ;
	logic       reset                   ;
	logic       accumulation       = 'b0;
	logic       plus_one           = 'b0;
	logic       clear_accu_control = 'b0;
	logic       dataflow_in        = 'b0;
	logic       control1           = 'b0;
	logic       update_w           = 'b0;
	reg         mac_en             = 'b0;
	wire        result                  ;
	wire        input_accu_adder        ;
	reg   [7:0] cur_weight              ;
	int         time_cnt           = 0  ;
	reg   [1:0] mac_en_dly         = 'b0;

	j_mac i_mac (
		.clk               (clk             ),
		.reset             (reset           ),
		.accumulation      (accumulation    ),
		.plus_one          (plus_one        ),
		.clear_accu_control(plus_one        ),
		.dataflow_in       (dataflow_in     ),
		.control1          (control1        ),
		.update_w          (update_w        ),
		.mac_en            (mac_en          ),
		.result            (result          ),
		.input_accu_adder  (input_accu_adder)
	);



	// always @(posedge clk) begin : proc_mac_en_dly
	// 	if(reset) begin
	// 		mac_en_dly <= 0;
	// 	end else begin
	// 		mac_en_dly <= #1 mac_en;
	// 	end
	// end

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

	task automatic put_clear_acc();
		clear_accu_control = 1'b1;
		@(posedge clk);
		clear_accu_control = 1'b0;
	endtask : put_clear_acc	

	task automatic put_data_i(input data, input is_update_w);
		update_w    = is_update_w;
		dataflow_in = data;
		@(posedge clk);
	endtask : put_data_i

	task automatic put_acc_i(input acc_in);
		accumulation    = acc_in;
		@(posedge clk);
	endtask : put_acc_i	

	task automatic put_plus_one(input one);
		plus_one    = one;
		@(posedge clk);
	endtask : put_plus_one

	task automatic put_control1(input control1_i);
		control1    = control1_i;
		// @(posedge clk);
	endtask : put_control1

	task automatic get_result_bit(output logic result_o);
		@(posedge clk);
		result_o = result;
		//$display($stime, "timer = %d, result_o = %b", time_cnt, result_o);
	endtask : get_result_bit

	task automatic put_mac_en(input mac_en_i);
		mac_en     = mac_en_i;
		@(posedge clk);
	endtask : put_mac_en


	always @(posedge clk) begin : proc_mac_en_dly
		if(reset)
			mac_en_dly <= 0;
		else
			mac_en_dly <= {mac_en_dly[0], mac_en};
	end

	// initial begin
	// 	forever begin
	// 		#1;
	// 		mac_en_dly = mac_en;
	// 		@(posedge clk);
	// 	end
	// end

	// parallel part
	task automatic shift_data_in(input logic [7:0] data);
		for (int i = 0; i < 8; i++) begin
			put_data_i(.data(data[i]), .is_update_w(1'b0));
		end
		for (int i = 8; i < 32; i++) begin
			put_data_i(.data(1'b0), .is_update_w(1'b0));
		end
	endtask : shift_data_in

	task automatic shift_weight_in(input int weight);
		reg [7:0] weight_abs;
		weight_abs = weight < 0 ? -weight : weight;
		put_control1(.control1_i(weight>=0 ? 1'b1 : 1'b0));
		for (int i = 0; i < 8; i++) begin
			put_data_i(.data(weight_abs[i]), .is_update_w(1'b1));
		end
	endtask : shift_weight_in

	task automatic shift_acc_in(input logic [31:0] acc_in);
		for (int i = 0; i < 32; i++) begin
			put_acc_i(.acc_in(acc_in[i]));
		end
	endtask : shift_acc_in

	task automatic shift_mac_en();
		time_cnt = 0;
		for (int i = 0; i < 32; i++) begin
			put_mac_en(.mac_en_i(1'b1));
			time_cnt = time_cnt + 1;
		end
		// put_mac_en(.mac_en_i(1'b0));
		//mac_en = 1'b0;
	endtask : shift_mac_en

	task automatic shift_one_in();
		for (int i = 0; i < 32; i++) begin
			if(i == 0)                    put_plus_one(.one(1'b1));
			else                          put_plus_one(.one(1'b0));
		end
	endtask : shift_one_in

	task automatic get_result(output logic [31:0] result_o);
		//get_clk();
		for (int i = 0; i < 32; i = (mac_en_dly[1]==1 ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i]));
			//$display($stime, " get result bit %d = %b", i, result_o[i]);
		end
	endtask : get_result

	task automatic put_clr_acc();
		//get_clk();
		for (int i = 0; i < 32; i++) begin
			if(i==31) put_clear_acc();
			else      get_clk();
		end
	endtask : put_clr_acc

	task automatic check_32(input logic [31:0] golden, input logic [31:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden %d != got %d", golden, check_result);
			$finish;
		end
		else begin
			$display($stime, " (golden) %d == (result) %d", golden, check_result);
		end
	endtask : check_32


	int check_queue[$];
	int result_queue[$];

	initial begin
		int  check_result = 0;
		int  new_w;
		int  new_data;
		int  new_acc = 0;
		logic [7:0] low_w;
		logic [7:0] low_data;

		#100;
		get_clk();

		fork
			begin
				forever begin
					get_result(check_result);
					result_queue.push_back(check_result);
				end
			end
			begin
				forever begin
					if((result_queue.size() != 0) && (check_queue.size() != 0)) begin
						int pop_result;
						int pop_check;
						pop_result = result_queue.pop_front();
						pop_check  = check_queue.pop_front();
						//$display($stime, " pop check=%d, result=%d", pop_check, pop_result);
						check_32(pop_check, pop_result);
					end else begin
						get_clk;
					end
				end
			end
		join_none


		for(new_w=-10; new_w<10; new_w++) begin
			put_mac_en(.mac_en_i(1'b0));
			shift_weight_in(.weight(new_w));
			for(new_acc=-10; new_acc<10; new_acc++) begin
				for(new_data=0; new_data<10; new_data++) begin
					int golden_mac;
					low_data = new_data;
					//$display($stime, " low_w=0xb%b, low_data=0xb%b, new_w=%d, new_data=%d", low_w, low_data, new_w, new_data);

					fork
						shift_data_in(.data(low_data));
						shift_mac_en();
						shift_one_in();
						shift_acc_in(.acc_in(new_acc));
						// get_result(.result_o(check_result));
						// put_clr_acc();
					join
					$display(" (%4d * %4d) + %4d = %4d", new_data, new_w, new_acc, (new_data*new_w) + new_acc);
					golden_mac = (new_data*new_w) + new_acc;
					//$display($stime, " push check=%d", golden_mac);
					check_queue.push_back(golden_mac);
					//check_32((new_data*new_w) + new_acc, check_result);
				end
			end
		end

		get_clk();
		$finish;
	end

endmodule