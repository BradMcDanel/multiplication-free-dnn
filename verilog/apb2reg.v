// $Id$
// This file was generated by regmaker.pl

module apb2reg (
    input              clk           ,
    input              reset_n       ,
    // APB Master
    input              psel          ,
    input       [15:2] paddr         ,
    input              pwrite        ,
    input       [31:0] pwdata        ,
    input              penable       ,
    output      [31:0] prdata        ,
    output             pready        ,
    // reg_ctrl
    output wire [31:0] reg_write_data,
    output wire [15:0] reg_addr      ,
    input  wire [31:0] reg_read_data ,
    output wire        reg_write     ,
    output             reg_read      ,
    output             reg_idle
);

    reg        reg_wr_en         ;
    reg        reg_re_en         ;
    reg        reg_wr_en_dly     ;
    reg        reg_re_en_dly     ;
    reg        reg_idle_reg      ;
    reg [31:0] reg_read_data_dly ;
    reg [31:0] reg_write_data_dly;
    reg [15:0] reg_addr_dly      ;


    always @(posedge clk) begin : proc_reg_write_data_dly
        if(~reset_n) begin
            reg_write_data_dly <= 0;
        end else if(psel) begin
            reg_write_data_dly <= pwdata;
        end
    end

    always @(posedge clk) begin : proc_reg_addr_dly
        if(~reset_n) begin
            reg_addr_dly <= 0;
        end else if(psel) begin
            reg_addr_dly <= {paddr, 2'b0};
        end
    end


    always @(posedge clk) begin : proc_reg_wr_en
        if(~reset_n) begin
            reg_wr_en <= 0;
        end else if(reg_wr_en) begin
            reg_wr_en <= 1'b0;
        end else if(psel & pwrite & ~penable) begin
            reg_wr_en <= 1'b1;
        end
    end

    always @(posedge clk) begin : proc_reg_re_en
        if(~reset_n) begin
            reg_re_en <= 0;
        end else if(reg_re_en)begin
            reg_re_en <= 1'b0;
        end else if(psel & ~pwrite & ~penable) begin
            reg_re_en <= 1'b1;
        end
    end

    always @(posedge clk) begin : proc_reg_re_en_dly
        if(~reset_n) begin
            reg_re_en_dly <= 0;
        end else begin
            reg_re_en_dly <= reg_re_en;
        end
    end


    always @(posedge clk) begin : proc_reg_wr_en_dly
        if(~reset_n) begin
            reg_wr_en_dly <= 0;
        end else begin
            reg_wr_en_dly <= reg_wr_en;
        end
    end

    always @(posedge clk) begin : proc_reg_read_data_dly
        if(~reset_n) begin
            reg_read_data_dly <= 0;
        end else begin
            reg_read_data_dly <= reg_read_data;
        end
    end

    always @(posedge clk) begin : proc_reg_idle_reg
        if(~reset_n) begin
            reg_idle_reg <= 1'b1;
        end else begin
            reg_idle_reg <= pready;
        end
    end

    assign reg_write      = reg_wr_en;
    assign reg_read       = reg_re_en;
    assign reg_addr       = reg_addr_dly;
    assign reg_write_data = reg_write_data_dly;
    assign reg_idle       = reg_idle_reg;

    assign prdata = reg_read_data_dly;
    assign pready = ~penable | (~reg_re_en & ~reg_re_en_dly) | (~reg_wr_en & reg_wr_en_dly);


endmodule
