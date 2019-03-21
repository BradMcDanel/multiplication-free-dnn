module save_to_bram_tb();


	logic               clk                                                                                     ;
	logic [(8*4*8-1):0] data_in         = 'b0                                                                   ;
	logic               saving_mode     = 'b0                                                                   ;
	logic [       17:0] user_input_addr = 18'b0                                                                 ;
	logic [   18*8-1:0] starting_addr   = {18'd7000,18'd6000,18'd5000,18'd4000,18'd3000,18'd2000,18'd1000,18'd0};
	wire  [        7:0] output_array                                                                            ;
	wire  [       17:0] sram_addr_ff                                                                            ;
	wire  [        7:0] sram_in_0                                                                               ;

	merged_wrapper tt (
		.clk_0            (clk            ),
		.saving_mode      (saving_mode    ),
		.data_in_0        (data_in        ),
		.data_o_0         (output_array   ),
		.user_input_addr_0(user_input_addr), // is the address that user want to access
		.starting_addr_0  (starting_addr  ), // is the starting addr of each of 8 channels
		.sram_in_0        (sram_in_0       ), // is the user access bram output
		.sram_addr_ff_0   (sram_addr_ff   )
	);


	initial begin
		$dumpfile("test.vcd");
		$dumpvars;
	end

	initial begin
		clk = 1;
		forever #5 clk = ~clk;
	end

	// serial part
	task automatic get_clk();
		@(posedge clk);
	endtask : get_clk

	//---------------------------------------------------------------------------------------------------------
	task automatic enter_saving_mode(input one);
		saving_mode    = one;
		@(posedge clk);
	endtask : enter_saving_mode

	task automatic leave_saving_mode(input zero);
		saving_mode    = zero;
		@(posedge clk);
	endtask : leave_saving_mode

	//task automatic shift_reset();
	//	for (int i = 0; i < 32; i++) begin
	//		if(i == 0)                    put_reset(.one(1'b1));
	//		else                          put_reset(.one(1'b0));
	//	end
	//endtask : shift_reset

	task automatic shift_32_data_in(input logic [7:0] data, input int pos_2, input int last_long);
		for (int i = 0; i < 32; i++) begin
			data_in[pos_2] = data[7];
			data_in[pos_2-1] = data[6];
			data_in[pos_2-2] = data[5];
			data_in[pos_2-3] = data[4];
			data_in[pos_2-4] = data[3];
			data_in[pos_2-5] = data[2];
			data_in[pos_2-6] = data[1];
			data_in[pos_2-7] = data[0];
			data = data + 4;
			if (i<31) begin
				#320;
			end
			if ((last_long) && (i == 31)) begin
				#320;
			end
		end
		@(posedge clk);
	endtask : shift_32_data_in

	task dummy_task ();
		for (int i = 0; i < 8; i++) begin
			saving_mode = 1;
		end
		@(posedge clk);
	endtask

	task automatic put_32_data_i(input data);
		data_in = data;
		@(posedge clk);
	endtask : put_32_data_i
	
    
    task automatic load_content_from_memory(input logic [17:0] start_addr);
        for (int i = 0; i < 50; i++) begin
            user_input_addr = start_addr + i;
            #10;
            $display($stime, " memory value at address %d is %d", user_input_addr, output_array);
        end
        
    endtask : load_content_from_memory
    
	task automatic check_8(input logic [7:0] golden, input logic [7:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden %d != got %d", golden, check_result);
			$finish;
		end
		else begin
			$display($stime, " (golden) %d == (result) %d", golden, check_result);
		end
	endtask : check_8


	int check_queue [$]    ;
	int result_queue[$]    ;
	int huhu            = 0;

	initial begin
		int         new_data;
		logic [7:0] low_w   ;
		logic [7:0] low_data;

		#100;
		get_clk();
		enter_saving_mode(.one(1));
		begin
			for (int i = 0; i < 4; i ++) begin
				for (int j = 0; j < 8 ; j ++) begin
					if (i == 3 && j == 7) begin   // for the last one only input once, do not last for 32 cycles
						fork
							begin
								shift_32_data_in(.data(i),.pos_2(32*j+7+8*i),.last_long(0));
								//#310;
							end

						join
						end
						else begin
							fork
								begin
									shift_32_data_in(.data(i),.pos_2(32*j+7+8*i),.last_long(1));
									//#310;
								end
								begin
									#10;
								end
							join_any
						end
						end
						end
						end
						enter_saving_mode(.one(0));

						// then loading data from memory to test
						load_content_from_memory(.start_addr(starting_addr[17:0]));
						load_content_from_memory(.start_addr(starting_addr[35:18]));
						load_content_from_memory(.start_addr(starting_addr[53:36]));
						load_content_from_memory(.start_addr(starting_addr[71:54]));
						load_content_from_memory(.start_addr(starting_addr[89:72]));
						load_content_from_memory(.start_addr(starting_addr[107:90]));
						load_content_from_memory(.start_addr(starting_addr[125:108]));
						load_content_from_memory(.start_addr(starting_addr[143:126]));
						get_clk();
						$finish;
						end
						endmodule

