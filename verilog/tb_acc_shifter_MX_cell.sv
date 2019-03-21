module tb_acc_shifter_MX_cell;

	parameter IMG_SIZE    = 32               ;
	parameter SRAM_DEPTH  = 32*32            ;
	parameter SRAM_ADDR_W = clog2(SRAM_DEPTH);

function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

	logic                         clk           = 'b0                     ;
	logic                         reset_n       = 'b0                     ;
	logic [                 31:0] sram_data     = 'b0                     ;
	logic                         shift_start   = 'b0                     ;
	logic [             1*32-1:0] shift_ctrl    = 'b0                     ;
	logic [   SRAM_ADDR_W*32-1:0] start_addr    = {(SRAM_ADDR_W*32){1'b0}};
	logic [      SRAM_ADDR_W-1:0] img_size      = 'b0                     ;
	wire                          sram_en                                 ;
	wire  [clog2(SRAM_DEPTH)-1:0] sram_addr                               ;
	wire                          shift_idle                              ;
	wire  [             1*32-1:0] serial_output                           ;
	wire  [             1*32-1:0] serial_en                               ;
	wire  [             1*32-1:0] serial_start                            ;

	logic                         we    ;
	logic [clog2(SRAM_DEPTH)-1:0] addr  ;
	logic [                  8:0] data_i;
	wire  [                  8:0] data_o;

	sram #(.DATA_WIDTH(32), .ADDR_WIDTH(clog2(SRAM_DEPTH)), .RAM_SIZE(SRAM_DEPTH)) i_sram (
		.clk   (clk      ),
		.we    (1'b0     ),
		.en    (sram_en  ),
		.addr  (sram_addr),
		.data_i(9'b0     ),
		.data_o(sram_data)
	);

	j_acc_shifter_MX_cell #(.SRAM_DEPTH(SRAM_DEPTH)) i_j_acc_shifter_MX_cell (
		.clk          (clk          ),
		.reset_n      (reset_n      ),
		.sram_en      (sram_en      ),
		.sram_addr    (sram_addr    ),
		.sram_data    (sram_data    ),
		.shift_start  (shift_start  ),
		.shift_idle   (shift_idle   ),
		.shift_ctrl   (shift_ctrl   ),
		.start_addr   (start_addr   ),
		.img_size     (img_size     ),
		.serial_output(serial_output),
		.serial_en    (serial_en    ),
		.serial_start (serial_start )
	);

	initial begin
	  $dumpfile("test.vcd");
	  $dumpvars;
	end

	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

	parameter SHT_KEEP = 0;
	parameter SHT_ZERO = 1;

	initial begin
		reset_n = 1'b1;
		@(posedge clk);
		reset_n = 1'b0;
		@(posedge clk);
		@(posedge clk);
		reset_n = 1'b1;
	end

	logic [31:0] result_queue[32][$];
	logic [31:0] check_queue [32][$];

	// serial part
	task automatic get_clk();
		@(posedge clk);
	endtask : get_clk

	task automatic get_result_bit(output logic result_o, input int idx);
		@(posedge clk);
		result_o = serial_output[idx];
	endtask : get_result_bit

	task automatic get_result(output logic [31:0] result_o, input int idx);
		for (int i = 0; i < 32; i = ((serial_en[idx]==1) ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i]), .idx(idx));
		end
	endtask : get_result

	task automatic check_32(input logic [31:0] golden, input logic [31:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden 0x%h != got 0x%h", golden, check_result);
			$finish;
		end
	endtask : check_32

	task automatic generate_check(input logic ctrl, input logic [31:0] start_addr, input int idx_c);
		logic        shift_ctrl;
		logic [31:0] cur_addr;

		shift_ctrl = ctrl;
		case(shift_ctrl)
			SHT_KEEP: begin
				cur_addr = start_addr;
				for (int cur_idx = 0; cur_idx <= img_size; cur_idx++) begin
					$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][31:0]);
					check_queue[idx_c].push_back(i_sram.RAM[cur_addr][31:0]);
					cur_addr ++;
				end
			end

			SHT_ZERO: begin
				cur_addr = start_addr;
				for (int cur_idx = 0; cur_idx <= img_size; cur_idx++) begin
					$display($stime, " push addr[%h] = %h", cur_addr, 0);
					check_queue[idx_c].push_back(0);
				end
			end
		endcase // shift_ctrl
	endtask


	initial begin
		int cur_idx;
		int cur_w;
		int cur_h;
		int cur_addr;
		logic pre_zero_skip;
		repeat(5) get_clk;
		
		start_addr = 0;
		img_size  = IMG_SIZE-1;


		`define FORK_RESULT_Q(idx_c) \
		fork \
			begin \
				forever begin \
					automatic logic [31:0]  check_result = 0; \
					get_result(.result_o(check_result), .idx(idx_c)); \
					$display($stime, " get   res[%d] <= ox%h", idx_c, check_result); \
					result_queue[idx_c].push_back(check_result); \
				end \
			end \
		join_none

		`FORK_RESULT_Q(00)
		`FORK_RESULT_Q(01)
		`FORK_RESULT_Q(02)
		`FORK_RESULT_Q(03)
		`FORK_RESULT_Q(04)
		`FORK_RESULT_Q(05)
		`FORK_RESULT_Q(06)
		`FORK_RESULT_Q(07)
		`FORK_RESULT_Q(08)
		`FORK_RESULT_Q(09)
		`FORK_RESULT_Q(10)
		`FORK_RESULT_Q(11)
		`FORK_RESULT_Q(12)
		`FORK_RESULT_Q(13)
		`FORK_RESULT_Q(14)
		`FORK_RESULT_Q(15)
		`FORK_RESULT_Q(16)
		`FORK_RESULT_Q(17)
		`FORK_RESULT_Q(18)
		`FORK_RESULT_Q(19)
		`FORK_RESULT_Q(20)
		`FORK_RESULT_Q(21)
		`FORK_RESULT_Q(22)
		`FORK_RESULT_Q(23)
		`FORK_RESULT_Q(24)
		`FORK_RESULT_Q(25)
		`FORK_RESULT_Q(26)
		`FORK_RESULT_Q(27)
		`FORK_RESULT_Q(28)
		`FORK_RESULT_Q(29)
		`FORK_RESULT_Q(30)
		`FORK_RESULT_Q(31)

		`define FORK_CHECK_Q(idx_c) \
		fork \
			begin \
				forever begin \
					int cur_idx; \
					cur_idx = 0; \
					if((result_queue[idx_c].size() != 0) && (check_queue[idx_c].size() != 0)) begin \
						automatic logic [31:0] pop_result; \
						automatic logic [31:0] pop_check; \
						pop_result = result_queue[idx_c].pop_front(); \
						pop_check  = check_queue[idx_c] .pop_front(); \
						$display($stime, " pop check=0x%h, result=0x%h for [%d][%d]", pop_check, pop_result, idx_c, cur_idx); \
						check_32(pop_check, pop_result); \
					end else if((result_queue[idx_c].size() != 0) && (check_queue[idx_c].size() == 0)) begin \
						$error($stime, " Error get more result @channel %d", idx_c); \
						$finish; \
					end else begin \
						get_clk; \
					end \
				end \
			end \
		join_none

		`FORK_CHECK_Q(00)
		`FORK_CHECK_Q(01)
		`FORK_CHECK_Q(02)
		`FORK_CHECK_Q(03)
		`FORK_CHECK_Q(04)
		`FORK_CHECK_Q(05)
		`FORK_CHECK_Q(06)
		`FORK_CHECK_Q(07)
		`FORK_CHECK_Q(08)
		`FORK_CHECK_Q(09)
		`FORK_CHECK_Q(10)
		`FORK_CHECK_Q(11)
		`FORK_CHECK_Q(12)
		`FORK_CHECK_Q(13)
		`FORK_CHECK_Q(14)
		`FORK_CHECK_Q(15)
		`FORK_CHECK_Q(16)
		`FORK_CHECK_Q(17)
		`FORK_CHECK_Q(18)
		`FORK_CHECK_Q(19)
		`FORK_CHECK_Q(20)
		`FORK_CHECK_Q(21)
		`FORK_CHECK_Q(22)
		`FORK_CHECK_Q(23)
		`FORK_CHECK_Q(24)
		`FORK_CHECK_Q(25)
		`FORK_CHECK_Q(26)
		`FORK_CHECK_Q(27)
		`FORK_CHECK_Q(28)
		`FORK_CHECK_Q(29)
		`FORK_CHECK_Q(30)
		`FORK_CHECK_Q(31)


		for (int i = 0; i <= img_size; i++) begin
			logic [31:0] x;
			x = i;
			i_sram.RAM[start_addr+i][31:0]     = x;
		end


		// for (int ctrl = SHT_KEEP; ctrl <=SHT_LEFT_UP ; ctrl++) begin
		for (int idx_c=0; idx_c<32; idx_c++) begin
			int ctrl;
			std::randomize (ctrl) with  {(ctrl >= SHT_KEEP) && (ctrl <= SHT_ZERO);};
			shift_ctrl[idx_c] = ctrl;
			$display($stime, " Generate check for channel %d, ctrl=%d", idx_c, ctrl);
			generate_check(.ctrl(ctrl), .start_addr(start_addr), .idx_c(idx_c));
			$display($stime, "===============================================");
		end

		for (int idx_c=0; idx_c<32; idx_c++) begin
			$display($stime, " check_queue %d size = %d", idx_c, check_queue[idx_c].size());
		end


		get_clk;
		#1;
		shift_start = 1;
		get_clk;
		#1;
		shift_start = 0;

		repeat(2) get_clk;
		while(~shift_idle) get_clk;

		repeat(500) get_clk;

		foreach(check_queue[idx_c]) begin
			if(check_queue[idx_c].size()!=0) begin
				$error($stime, " Error check queue[%d] is not empty", idx_c);
				$finish;
			end
		end

		$finish;
	end
endmodule