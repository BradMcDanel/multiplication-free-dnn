module tb_acc_deshifter;

	parameter IMG_SIZE = 10;
	parameter SRAM_DEPTH = 32*32;

function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

	logic                         clk         = 'b0;
	logic                         reset_n     = 'b0;
	wire  [                 31:0] sram_data        ;
	logic                         shift_start = 'b0;
	logic [clog2(SRAM_DEPTH)-1:0] start_addr  = 'b0;
	logic [clog2(SRAM_DEPTH)-1:0] img_size    = 'b0;

	wire                         sram_en  ;
	wire [clog2(SRAM_DEPTH)-1:0] sram_addr;

	wire  shift_idle         ;
	logic serial_input = 1'b0;
	logic serial_en    = 1'b0;


	logic                         we    ;
	logic [clog2(SRAM_DEPTH)-1:0] addr  ;
	logic [                 31:0] data_i;
	wire  [                 31:0] data_o;

	sram #(.DATA_WIDTH(32), .ADDR_WIDTH(clog2(SRAM_DEPTH)), .RAM_SIZE(SRAM_DEPTH)) i_sram (
		.clk   (clk      ),
		.we    (sram_en  ),
		.en    (sram_en  ),
		.addr  (sram_addr),
		.data_i(sram_data),
		.data_o(         )
	);

	j_acc_deshifter #(.SRAM_DEPTH(SRAM_DEPTH)) i_j_acc_deshifter (
		.clk         (clk         ),
		.reset_n     (reset_n     ),
		.sram_en     (sram_en     ),
		.sram_addr   (sram_addr   ),
		.sram_data   (sram_data   ),
		.shift_start (shift_start ),
		.shift_idle  (shift_idle  ),
		.start_addr  (start_addr  ),
		.img_size    (img_size    ),
		.serial_input(serial_input),
		.serial_en   (serial_en   )
	);

	initial begin
	  $dumpfile("test.vcd");
	  $dumpvars;
	end

	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

    parameter SHT_KEEP       = 0;
    parameter SHT_ZERO       = 1;
  
	initial begin
		reset_n = 1'b1;
		@(posedge clk);
		reset_n = 1'b0;
		@(posedge clk);
		@(posedge clk);
		reset_n = 1'b1;
	end

	logic [31:0] result_queue[$];
	logic [31:0] check_queue [$];
	logic [31:0] shift_queue [$];

	// serial part
	task automatic get_clk();
		@(posedge clk);
	endtask : get_clk

	// task automatic get_result_bit(output logic result_o);
	// 	@(posedge clk);
	// 	result_o = serial_output;
	// endtask : get_result_bit

	task automatic get_result(output logic [31:0] result_o);
		while(~sram_en) get_clk;
		result_o = sram_data;
		get_clk;
	endtask : get_result

	task automatic put_acc_i(input acc_in);
		serial_input    = acc_in;
		serial_en       = 1'b1;
		@(posedge clk);
	endtask : put_acc_i	

	task automatic shift_acc_in(input logic [31:0] acc_in);
		for (int i = 0; i < 32; i++) begin
			put_acc_i(.acc_in(acc_in[i]));
		end
	endtask : shift_acc_in

	task automatic check_32(input logic [31:0] golden, input logic [31:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden 0x%h != got 0x%h", golden, check_result);
			$finish;
		end
	endtask : check_32

	initial begin
		int cur_idx;
		int cur_addr;
		repeat(5) get_clk;
		
		start_addr = 0;
		img_size  = IMG_SIZE-1;

		fork 
			begin 
				forever begin 
					automatic logic [31:0]  check_result = 0; 
					get_result(check_result); 
					$display($stime, " get   res <= ox%h", check_result); 
					result_queue.push_back(check_result); 
				end 
			end 
		join_none 

		fork
			begin
				forever begin
					int cur_idx;
					cur_idx = 0;
					if((result_queue.size() != 0) && (check_queue.size() != 0)) begin
						automatic logic [31:0] pop_result;
						automatic logic [31:0] pop_check;
						pop_result = result_queue.pop_front();
						pop_check  = check_queue .pop_front();
						$display($stime, " pop check=0x%h, result=0x%h for [%d]", pop_check, pop_result, cur_idx);
						check_32(pop_check, pop_result);
					end else if((result_queue.size() != 0) && (check_queue.size() == 0)) begin
						$error($stime, " Error get more result");
						$finish;
					end else begin
						get_clk;
					end
				end
			end
		join_none

		fork 
			begin 
				forever begin 
					if((shift_queue.size() != 0) & ~shift_idle) begin
						automatic logic [31:0]  shift_acc = 0; 
						shift_acc = shift_queue.pop_front();
						shift_acc_in(shift_acc);
					end else begin
						#1
						serial_en = 1'b0;
						get_clk;
					end 
				end 
			end 
		join_none 


		for (int i = 0; i <= img_size; i++) begin
			logic [31:0] x;
			x = i;
			check_queue.push_back(x);
			shift_queue.push_back(x);
		end



		get_clk;
		#1;
		shift_start = 1;
		get_clk;
		#1;
		shift_start = 0;

		while(~shift_idle) get_clk;

		repeat(500) get_clk;
		$finish;
	end


final begin
	if(check_queue.size()!=0)
		$display($stime, " Error check queue is not empty");
end


endmodule